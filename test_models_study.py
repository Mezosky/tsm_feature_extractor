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


# No sé que hace esto, pero al parecer se utiliza
# para calcular metricas
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


# Aquí comienza el código con procesos de carga de pesos en la net
# notar que esto no es una función, pero posteriormente existen funciones
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


data_iter_list = []
net_list = []
modality_list = []

total_num = None
for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'RGB' in this_weights:
        modality = 'RGB'
    else:
        modality = 'Flow'
    this_arch = this_weights.split('TSM_')[1].split('_')[2]
    modality_list.append(modality)
    num_class, _, _, _, _ = dataset_config.return_dataset(args.dataset, modality)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
    net = TSN(num_class, this_test_segments if is_shift else 1, modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              )

    if 'tpool' in this_weights:
        from ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

    checkpoint = torch.load(this_weights)
    #ipdb.set_trace()
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
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
            TSNDataSetVideos(test_file, 
                            num_segments=this_test_segments,
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

    print("videos cargados")
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    print("1")
    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    print("2")
    data_gen = enumerate(data_loader)

    print("3")
    if total_num is None:
        total_num = len(data_loader.dataset)
    else:
        assert total_num == len(data_loader.dataset)

    print("4")
    data_iter_list.append(data_gen)
    print("5")
    net_list.append(net)
    print("6")
    #ipdb.set_trace()

# esta parte obtiene features, pero hay que notar que aplica softmax
# para obtener la label que esta clasificando. 
# Esta función se debería reutilizar pero quitar partes de evaluación
# o enfocadas en la clasificación
def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():
        # problema, el video_data posee una label, quizas debería poseer
        # una label ficticia que se rellene para que el código funcione solamente,
        # o netamente sacar esta parte y generar otro dataloader
        i, data, label = video_data # quitar la label del dataloader
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
        
        if is_shift:
             data_in = data_in.view(-1, this_test_segments, length, data_in.size(2), data_in.size(3))
        features, rst = net(data_in)
        
        return features

proc_start_time = time.time()
# No sé lo que hace, pero tiene que ver con uno de los agumentos
max_num = args.max_num if args.max_num > 0 else total_num

# Se utiliza un iterador de la data para recorrer los videos (o eso creo)
for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    with torch.no_grad():
        if i >= max_num:
            break
        features_list = []
        # aquí es solo necesario net_list, modality_list y una parte del data_label_pairs
        for n_seg, (_, (data, label)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
            # Aquí se llama a la función de arriba, hay que sacarla
            
            feature = eval_video((i, data, label), net, n_seg, modality)

            # notar que lo anterior entregaba 3 cosas, esto hay que modificarlo a uno
            features_list.append(feature)

        ipdb.set_trace()