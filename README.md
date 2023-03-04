# Feature extractor for TSN and TSM: 

## Original Papers
- Temporal Shift Module for Efficient Video Understanding [[Website]](https://hanlab.mit.edu/projects/tsm/) [[arXiv]](https://arxiv.org/abs/1811.08383)[[Demo]](https://www.youtube.com/watch?v=0T6u7S_gq-4)

## How to run

```shell
python feature_extractor_2.py kinetics --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth --test_segments=8 --test_crops=1     --batch_size=1 --root_path=/data/imeza/youcookii/without_subfolders/videos/ --num_frame=8 --output_path=/data/imeza/youcookii/ycii_features/TSM_TEST
```

