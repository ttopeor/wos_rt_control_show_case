import os
import cv2
import numpy as np
from object_segments.detect_obj import ObjDetectorSOLOv2
import pyrealsense2 as rs
from rs_mount.rs_mount_trans import from_camera_target_to_ee_target

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
REPO_DIR = os.path.join(BASE_DIR, '..')           
MODEL_DIR = os.path.join(REPO_DIR, 'model')
config_file = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco.py')
checkpoint_file  = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth')

detector = ObjDetectorSOLOv2(
    config_file=config_file,
    checkpoint_file=checkpoint_file,
    target_class='orange',
    score_thr=0.5
)

pipeline = rs.pipeline()
align_to = rs.stream.color
align = rs.align(align_to)

config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Get depth scale (meters per depth unit)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = detector.get_object_xyz(
            color_image, 
            depth_image,
            depth_intrin=depth_intrin,
            depth_scale=depth_scale,
            max_depth=10000,  
            sigma_factor=2.0   
        )

        vis_img = color_image.copy()

        for i, obj_info in enumerate(results):
            mask = obj_info['mask']
            (x_c, y_c, z_c) = obj_info['center_3d']

            print(f"[Object {i}] 3D center = (x={x_c:.3f}, y={y_c:.3f}, z={z_c:.3f})")
            
            overlay = np.zeros_like(vis_img, dtype=np.uint8)
            overlay_color = (0, 255, 0) 
            overlay[mask] = overlay_color

            alpha = 0.5  
            vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

        cv2.imshow("Obj Detection (SOLOv2)", vis_img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
