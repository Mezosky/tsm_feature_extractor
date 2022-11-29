# test the TSN and TSM on Kinetics using 8-frame, you should get top-1 accuracy around:
# TSN: 68.8%
# TSM: 71.2%

# test TSN
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64

# test TSM
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64

# Feature extractor
python test_models_study.py kinetics \
--weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
--test_segments=8 --test_crops=1  --batch_size=1 --test_list=/data/imeza/ActivityNet/test_videos/

python feature_extractor.py kinetics \
--weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
--test_segments=8 --test_crops=1     --batch_size=1 --test_list=/data/imeza/ActivityNet/test_videos/

python feature_extractor_2.py kinetics \
--weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
--test_segments=8 --test_crops=1     --batch_size=1 --root_path=/data/imeza/ActivityNet/test_videos/ --num_frame=8