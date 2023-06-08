# Step1. Install ByteTrack.
pip3 install -r requirements.txt
python3 setup.py develop

# Step2. Install pycocotools.
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Step3. Others
pip3 install cython_bbox
pip3 install mmcv-full==1.7.1
pip3 install -v -e .
