import os
import argparse
import time
import numpy as np
import pyrealsense2 as rs
import cv2

from object_segments.detect_obj import ObjDetectorSOLOv2
from graspnet_baseline.obj_affordance_generator import ObjectAffordanceGenerator
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(REPO_DIR, 'model')

solov2_config_path = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco.py')
solov2_ckpt_path  = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth')
graspnet_ckpt_path = os.path.join(MODEL_DIR, 'checkpoint-rs.tar')

parser = argparse.ArgumentParser()
parser.add_argument('--graspnet_ckpt', type=str, default=graspnet_ckpt_path)
parser.add_argument('--num_point', type=int, default=20000)
parser.add_argument('--num_view',  type=int, default=300)
parser.add_argument('--collision_thresh', type=float, default=0.01)
parser.add_argument('--voxel_size', type=float, default=0.01)
parser.add_argument('--visualize', action='store_true', default=True)
args = parser.parse_args()

generator = ObjectAffordanceGenerator(
    graspnet_checkpoint_path=args.graspnet_ckpt,
    num_point=args.num_point,
    num_view=args.num_view,
    collision_thresh=args.collision_thresh,
    voxel_size=args.voxel_size,
    visualize=args.visualize
)

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is:", depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    detector = ObjDetectorSOLOv2(
        config_file=solov2_config_path,
        checkpoint_file=solov2_ckpt_path,
        target_class='cup',
        score_thr=0.5
    )

    try:
        while True:
            start_time = time.time()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            fx, fy = intr.fx, intr.fy
            cx, cy = intr.ppx, intr.ppy
            intrinsic = np.array([
                [fx,   0,  cx],
                [ 0,  fy,  cy],
                [ 0,   0,   1]
            ], dtype=np.float32)

            # Detect objects
            masks = detector.detect_obj(color_image)
            num_masks = len(masks)

            # If no object found
            if num_masks == 0:
                elapsed = time.time() - start_time
                print(f"No objects found. Time for this frame: {elapsed:.3f} s")
                continue

            # Combine all masks
            H, W = depth_image.shape
            mask_bool = np.zeros((H, W), dtype=bool)
            for m in masks:
                mask_bool |= m

            # Depth filtering
            cup_depths = depth_image[mask_bool & (depth_image > 0)]
            if len(cup_depths) > 0:
                d_mean = np.mean(cup_depths)
                d_std  = np.std(cup_depths)
                lower = d_mean - 2*d_std
                upper = d_mean + 2*d_std
                refined_mask = mask_bool & (depth_image >= lower) & (depth_image <= upper)
            else:
                refined_mask = np.zeros_like(mask_bool, dtype=bool)

            # GraspNet forward
            end_points, cloud_o3d = generator.process_data(
                color_image=color_image,
                depth_image=depth_image,
                workspace_mask=refined_mask,
                intrinsic=intrinsic,
                depth_scale=depth_scale
            )

            gg = generator.get_grasps(end_points)

            gg.nms()
            gg.sort_by_score()
            # Print some grasp info
            num_grasps = len(gg)
            if num_grasps == 0:
                elapsed = time.time() - start_time
                print(f"No valid grasps found! Time for this frame: {elapsed:.3f} s")
                continue
            
            num_grasps = len(gg)
            scores = [g.score for g in gg]
            avg_score = np.mean(scores)
            
            print(f"Detected {num_masks} object(s). Found {num_grasps} grasps. Avg score: {avg_score:.3f}")

            # Collision detection
            if args.collision_thresh > 0:
                gg = generator.collision_detection(gg, np.asarray(cloud_o3d.points, dtype=np.float32))
                print(f"After collision check: {len(gg)} grasps remain")
                
            num_grasps = len(gg)
            if num_grasps == 0:
                elapsed = time.time() - start_time
                print(f"No valid grasps After collision check!")
                continue

                
            best_grasp = gg[0]
            if best_grasp.score < 0.7:
                elapsed = time.time() - start_time
                print(f"Best grasp score {best_grasp.score:.3f} < 0.7, skip. Time: {elapsed:.3f} s")
                continue
            
            print("Best grasp score:", best_grasp.score)
            print("Translation (xyz):", best_grasp.translation)
            print("Rotation matrix:\n", best_grasp.rotation_matrix)
            print("Width:", best_grasp.width)
            print("Height:", best_grasp.height)
            print("Depth:", best_grasp.depth)
            print("Object ID:", best_grasp.object_id)

            # Visualize
            if args.visualize and len(gg) > 0:
                print("Press [q] in the Open3D window to continue...")
                generator.vis_grasps(gg, cloud_o3d, topK=1)

            elapsed = time.time() - start_time
            print(f"Time for this frame: {elapsed:.3f} s")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
