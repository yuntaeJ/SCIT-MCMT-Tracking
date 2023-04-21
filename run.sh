
# Single-camera Tracking on multicam

python3 tools/multicam_track.py video \
./test-site00 \
../experiments/yolox/yolox_x.py \
../experiments/yolox-clean/epoch_250.pth \
--tp_weight ../tp/tp_best.pth \
--save_result ./multicam-site00 --save_vid True --track_buffer 150 

