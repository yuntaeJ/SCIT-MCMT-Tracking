
# Single-camera Tracking on multicam

python3 tools/multicam_track.py video \
./test-site022 \
../experiments/yolox/yolox_x.py \
../experiments/yolox/epoch_250.pth \
--tp_weight ../tp/tp_best.pth \
--save_result ./test-site022 --save_vid True --track_buffer 150 
