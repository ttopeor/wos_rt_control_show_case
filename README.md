# wos_rt_control_show_case
python version = 3.9.20


pip install -e .

## take very long time
mim install mmcv==2.1.0

mkdir model
cd model
mim download mmdet --config solov2_r50_fpn_1x_coco --dest .

cd module/graspnet-baseline/
pip install .
cd pointnet2
python setup.py install
cd ..
cd knn
python setup.py install

cd ..
cd graspnetAPI
pip install .

download https://drive.google.com/file/d/1DcjGGhZIJsxd61719N0iWA7L6vNEK0ci/view
decompress and move it into module/graspnet-baseline/tolerance

download https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view
move tar to model/
