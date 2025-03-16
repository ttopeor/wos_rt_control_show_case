import os
import sys
import argparse
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import cv2
import torch

from graspnet_baseline.models.graspnet import GraspNet, pred_decode
from graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector
from graspnet_baseline.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup

from object_segments.detect_obj import ObjDetectorSOLOv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # test.py所在目录test/
REPO_DIR = os.path.join(BASE_DIR, '..')                # 上一级 即repo-root
MODEL_DIR = os.path.join(REPO_DIR, 'model')

parser = argparse.ArgumentParser()

parser.add_argument('--num_point', type=int, default=20000, 
                    help='Number of points sampled from point cloud [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, 
                    help='Number of views [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, 
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, 
                    help='Voxel Size for downsampling in collision detection [default: 0.01]')
parser.add_argument('--visualize', action='store_true', default=True,
                    help='Whether to visualize the final 3D grasps via Open3D')
cfgs = parser.parse_args()

graspnet_checkpoint_path =  os.path.join(MODEL_DIR, 'checkpoint-rs.tar')
solov2_config_path = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco.py')
solov2_ckpt_path  = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth')


def get_net():
    # 初始化 GraspNet 模型
    net = GraspNet(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(graspnet_checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"-> Loaded checkpoint {graspnet_checkpoint_path} (epoch: {start_epoch})")

    net.eval()
    return net


def process_data(color_image, depth_image, workspace_mask, intrinsic, depth_scale):
    """
    基于彩色、深度、工作区掩码、内参来生成 GraspNet 所需的输入:
       end_points: 字典，包括 [B, num_point, 3] 的采样点云 (GPU tensor) 等信息
       cloud_o3d:  open3d.geometry.PointCloud 对象，便于后续可视化
    """
    # 1) 转为浮点并归一化颜色到 [0,1]
    color = color_image.astype(np.float32) / 255.0
    depth = depth_image  # (H,W) uint16

    # 2) 构造 CameraInfo 并生成三维点云 (organized)
    H, W = depth.shape
    cam_info = CameraInfo(
        width=W, 
        height=H,
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2],
        scale=(1.0 / depth_scale)  
    )
    cloud = create_point_cloud_from_depth_image(depth, cam_info, organized=True)

    # 3) 只保留 workspace_mask & depth>0 的点
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]   # [N, 3]
    color_masked = color[mask]   # [N, 3]

    # 4) 若点数太多则随机下采样，否则重复采样
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs, :]
    color_sampled = color_masked[idxs, :]

    # 5) 构造 open3d 云对象(可视化用)
    import open3d as o3d
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # 6) 把采样点云放到 GPU Tensor
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)

    end_points = {
        'point_clouds': cloud_sampled,
        'cloud_colors': color_sampled  # 调试用
    }
    return end_points, cloud_o3d


def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()  # (num_grasp, 8)
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud_points):
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg_filtered = gg[~collision_mask]
    return gg_filtered


def vis_grasps(gg, cloud_o3d, topK=50):
    gg.nms()
    gg.sort_by_score()
    gg_topk = gg[:topK]
    grippers = gg_topk.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud_o3d, *grippers])


def main():
    # 1) 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # 获取深度刻度
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is:", depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # 2) 初始化 GraspNet
    net = get_net()

    # 3) 初始化 SOLOv2 水杯检测器
    detector = ObjDetectorSOLOv2(        
        config_file=solov2_config_path,
        checkpoint_file=solov2_ckpt_path,
        target_class='cup',
        score_thr=0.5)

    try:
        while True:
            # ========== 获取 RealSense 帧并对齐 ==========
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())  # (720,1280) uint16
            color_image = np.asanyarray(color_frame.get_data())  # (720,1280,3) uint8

            # 取出相机内参
            intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            fx, fy = intr.fx, intr.fy
            cx, cy = intr.ppx, intr.ppy
            intrinsic = np.array([
                [fx,   0,  cx],
                [ 0,  fy,  cy],
                [ 0,   0,   1]
            ], dtype=np.float32)

            # ========== 用 SOLOv2 分割检测水杯，得到掩码列表 ==========
            bboxes, masks = detector.detect_obj(color_image)
            if len(masks) == 0:
                # 没检测到任何杯子
                cv2.imshow('Color', color_image)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            # 这里示例：把所有水杯掩码做“或”运算，合并成一个总掩码
            # 如果你只想抓取第一个杯子，可直接 mask_bool = masks[0]
            H, W = depth_image.shape
            mask_bool = np.zeros((H, W), dtype=bool)
            for m in masks:
                mask_bool |= m  # 合并所有杯子区域

            cup_depths = depth_image[mask_bool & (depth_image > 0)]
            if len(cup_depths) > 0:
                d_mean = np.mean(cup_depths)
                d_std  = np.std(cup_depths)
                # 定义一个范围，比如 [mean - 2*std, mean + 2*std]
                lower = d_mean - 2*d_std
                upper = d_mean + 2*d_std

                # 二次过滤：只保留在此区间内的像素
                refined_mask = mask_bool & (depth_image >= lower) & (depth_image <= upper)
            else:
                # 如果掩码里全是无效深度，那么 refined_mask 就全 False
                refined_mask = np.zeros_like(mask_bool, dtype=bool)
    
            # ========== 调用 GraspNet 流程 ==========
            end_points, cloud_o3d = process_data(
                color_image=color_image,
                depth_image=depth_image,
                workspace_mask=refined_mask,
                intrinsic=intrinsic,
                depth_scale=depth_scale
            )

            gg = get_grasps(net, end_points)

            gg.nms()            # 可选：做一次 NMS 以去除相似抓取
            gg.sort_by_score()  # 从高到低按score排序
            
            if len(gg) == 0:
                print("No valid grasps found! gg is empty.")
                # 你可以选择 continue 跳过这一帧
                continue

            best_grasp = gg[0]
            if best_grasp.score < 1.0:
                continue
            
            print("Best grasp score:", best_grasp.score)
            print("Translation (xyz):", best_grasp.translation)   # ndarray, shape=(3,)
            print("Rotation matrix:\n", best_grasp.rotation_matrix)      # ndarray, shape=(3,3)
            print("Width:", best_grasp.width)
            print("Height:", best_grasp.height)
            print("Depth:", best_grasp.depth)
            print("Object ID:", best_grasp.object_id)

            # (可选) 碰撞检测
            if cfgs.collision_thresh > 0:
                gg = collision_detection(gg, np.asarray(cloud_o3d.points, dtype=np.float32))

            # (可选) 可视化 3D 抓取 (Open3D)
            if cfgs.visualize:
                print("Press [q] in the Open3D window to continue...")
                vis_grasps(gg, cloud_o3d, topK=1)

            # ========== 在 2D 窗口中简单可视化 ==========
            # 1) 在 color_image 上用红色覆盖所有 cup 掩码区域
            color_viz = color_image.copy()
            for m in masks:
                color_viz[m] = (0, 0, 255)  # 红色

            # 2) 在 color_image 上画 bounding box，仅作为调试
            for (x1, y1, x2, y2) in bboxes:
                cv2.rectangle(color_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Color', color_viz)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break


    finally:
        pipeline.stop()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
