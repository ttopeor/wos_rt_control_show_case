# wos_rt_control_show_case
python version = 3.9.20


pip install -e .

## take very long time
mim install mmcv

mkdir model & cd model
mim download mmdet --config solov2_r50_fpn_1x_coco --dest .

cd ..
pip install -r module/graspnet-baseline/requirements.txt
python module/graspnet-baseline/pointnet2/setup.py install
python module/graspnet-baseline/knn/setup.py install
pip install module/graspnetAPI

download https://drive.google.com/file/d/1DcjGGhZIJsxd61719N0iWA7L6vNEK0ci/view
decompress and move it into module/graspnet-baseline/tolerance

