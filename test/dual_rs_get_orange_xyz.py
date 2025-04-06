import os
import time
import cv2
import numpy as np

from object_segments.detect_obj import ObjDetectorSOLOv2
from utils.config_loader import load_config

from rt_loops.read_rs_frams_loop import ReadRSFrameLoop


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(REPO_DIR, 'model')


class OrangeGraspDemo:

    def __init__(self):
        # 1) Load configuration
        try:
            self.config = load_config()
        except FileNotFoundError as e:
            print("Configuration file not found:", e)
            raise

        front_rs_serial = self.config["rs_D405"]["front_camera_serial"]
        
        self.from_ee_to_cam_trans = self.config["rs_D405"]["mount_transfer_dual_front"]
        self.from_cam_to_ee_target_trans = self.config["rs_D405"]["target_transfer_dual_front"]
        
        self.rs_loop = ReadRSFrameLoop(freq=30, serial_number=front_rs_serial)  # Example: 30 FPS


        # 6) Initialize object detection and grasp prediction models
        solov2_config_path = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco.py')
        solov2_ckpt_path  = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth')

        self.obj_detector = ObjDetectorSOLOv2(
            config_file=solov2_config_path,
            checkpoint_file=solov2_ckpt_path,
            target_class='orange',
            score_thr=0.5
        )

        time.sleep(2)  # Allow time for models/threads to initialize


    def run(self):
        try:
            while True:                
                # (A) Retrieve the latest camera frames
                color_image, depth_image = self.rs_loop.get_latest_frames()
                if color_image is None or depth_image is None:
                    print("Waiting for camera frames...")
                    time.sleep(0.1)
                    continue

                camera_target = self.get_camera_target_from_solo_v2()
                if camera_target is None:
                    continue
                print(camera_target)


        except KeyboardInterrupt:
            print("[OrangeGraspDemo] Ctrl + C received. Exiting...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down...")

        # Stop background threads if required
        self.rs_loop.shutdown()

        print("Shutdown complete.")
        
    def get_camera_target_from_solo_v2(self):
        color_image, depth_image = self.rs_loop.get_latest_frames()
        if color_image is None or depth_image is None:
            print("[get_camera_target_from_solo_v2] No camera frames available.")
            return None

        depth_intrin = self.rs_loop.get_intrinsic()
        depth_scale = self.rs_loop.get_depth_scale()

        results = self.obj_detector.get_object_xyz(
            color_image,
            depth_image,
            depth_intrin=depth_intrin,
            depth_scale=depth_scale,
            max_depth=10000, 
            sigma_factor=2.0
        )

        vis_img = color_image.copy()

        if results:
            for i, obj_info in enumerate(results):
                mask = obj_info['mask']
                (x_c, y_c, z_c) = obj_info['center_3d']

                overlay = np.zeros_like(vis_img, dtype=np.uint8)
                overlay_color = (0, 255, 0) 
                overlay[mask] = overlay_color
                alpha = 0.5
                vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)
        else:
            cv2.imshow("SOLOv2 Mask Overlay", vis_img)
            cv2.waitKey(1)
            #print("[get_camera_target_from_solo_v2] No objects found by SOLOv2.")
            return None

        cv2.imshow("SOLOv2 Mask Overlay", vis_img)
        cv2.waitKey(1)

        if results:
            chosen = min(results, key=lambda obj: obj['center_3d'][2])
            (x_c, y_c, z_c) = chosen['center_3d']
            #print(f"[get_camera_target_from_solo_v2] Nearest object center = ({x_c:.3f}, {y_c:.3f}, {z_c:.3f})")
            return [-x_c, -y_c, z_c, 0, 0, 0] #color_image and depth_image are up/down left/right shifted
        else:
            return None
        
def main():
    demo = OrangeGraspDemo()
    demo.run()

if __name__ == "__main__":
    main()
