# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.dataset_video import TSNDataSetVideos
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F

import ipdb

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

args = parser.parse_args()


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

def eval_video(video_data, net, test_segments_list, modality):
    net.eval()

    with torch.no_grad():
        i, data, label = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        data_in = data.view(-1, length, data.size(2), data.size(3))
        batch_size = 2
        if is_shift:
             data_in = data_in.view(batch_size * num_crop, test_segments_list, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        return i, rst, label


weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)

is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weights_list[0])

if 'RGB' in weights_list[0]:
    modality = 'RGB'
else:
    modality = 'Flow'

this_arch = weights_list[0].split('TSM_')[1].split('_')[2]

num_class, _, _, _, _ = dataset_config.return_dataset(args.dataset, modality)
print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
net = TSN(num_class, test_segments_list[0] if is_shift else 1, modality,
            base_model=this_arch,
            consensus_type=args.crop_fusion_type,
            img_feature_dim=args.img_feature_dim,
            pretrain=args.pretrain,
            is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
            non_local='_nl' in weights_list[0],
            )

if 'tpool' in weights_list[0]:
    from ops.temporal_shift import make_temporal_pool
    make_temporal_pool(net.base_model, test_segments_list)  # since DataParallel

checkpoint = torch.load(weights_list[0])
checkpoint = checkpoint['state_dict']

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net.load_state_dict(base_dict)

input_size = net.scale_size if args.full_res else net.input_size
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 3:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 5:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSetVideos(   test_file_list[0], 
                            num_segments=test_segments_list[0],
                            new_length=1 if modality == "RGB" else 5,
                            modality=modality,
                            test_mode=True,
                            remove_missing=len(weights_list) == 1,
                            transform=torchvision.transforms.Compose([
                                cropping,
                                Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                                GroupNormalize(net.input_mean, net.input_std),
                            ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))
net = torch.nn.DataParallel(net.cuda())

# extraimos features
this_rst_list = []
for i, video in enumerate(data_loader):
    with torch.no_grad():
        rst = eval_video((i, video[0], video[1]), net, test_segments_list[0], modality)
        this_rst_list.append(rst[1])

ipdb.set_trace()