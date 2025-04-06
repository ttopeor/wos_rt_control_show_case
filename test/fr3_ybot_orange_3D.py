import os
import time
import cv2
import numpy as np

from object_segments.detect_obj import ObjDetectorSOLOv2
from utils.config_loader import load_config

from rt_loops.read_rs_frams_loop import ReadRSFrameLoop
from rt_loops.read_arm_pos_loop import ReadArmPositionLoop
from utils.math_tools import compute_6d_distance
from wos_api.robot_rt_control import robot_rt_control

from rs_mount.rs_mount_trans import from_camera_target_to_ee_target

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(REPO_DIR, 'model')


class OrangeGraspDemo:
    """
    A class that demonstrates object detection, affordance-based grasping prediction, 
    and robotic arm control for grasping a Orange. The main loop follows a step-by-step 
    process instead of running at a fixed frequency.
    """

    def __init__(self):
        # 1) Load configuration
        try:
            self.config = load_config()
        except FileNotFoundError as e:
            print("[OrangeGraspDemo] Configuration file not found:", e)
            raise

        self.fr3_home_pos = self.config["fr3_home_pos"]
        fr3_id = self.config["fr3_resource_id"]
        wos_endpoint = self.config["wos_end_point"]
        print("[OrangeGraspDemo] WOS Endpoint:", wos_endpoint)

        self.from_ee_to_cam_trans = self.config["rs_D405"]["mount_transfer_curve"]
        self.from_cam_to_ee_target_trans = self.config["rs_D405"]["target_transfer_curve"]
        
        # 2) Start the RealSense frame acquisition loop
        self.rs_loop = ReadRSFrameLoop()  # Example: 30 FPS
        # 3) Start the robotic arm position reading loop
        self.read_arm_position_loop = ReadArmPositionLoop(fr3_id, wos_endpoint)
        # Set frequency if the loop has a built-in mechanism
        self.read_arm_position_loop.loop_spin(frequency=50)

        # 4) Initialize robot control interface
        self.write_arm_position = robot_rt_control(fr3_id, wos_endpoint)

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
        
        # 5) Move the robotic arm to its home position
        self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 3)
        self.write_arm_position.open_fr3_gripper()
        print("[OrangeGraspDemo] Homing Arm fr3...")
        time.sleep(3)


        print("[OrangeGraspDemo] Setup completed.")
        self.start_time = time.time()
        self.grasp_count = 0
    def run(self):
        """
        Continuously executes the object detection, affordance estimation, 
        and robotic grasping process in a loop. 
        Exits on Ctrl+C.
        """
        try:
            while True:                
                # (A) Retrieve the latest camera frames
                color_image, depth_image = self.rs_loop.get_latest_frames()
                if color_image is None or depth_image is None:
                    print("[OrangeGraspDemo] Waiting for camera frames...")
                    time.sleep(0.1)
                    continue

                arm_pos_reading = self.read_arm_position_loop.get_position_reading()

                camera_target = self.get_camera_target_from_solo_v2()
                if camera_target is None:
                    obj_not_found_time = time.time() - self.start_time
                    distance = compute_6d_distance(arm_pos_reading, self.fr3_home_pos)
                    if distance >= 0.02 and obj_not_found_time > 2:
                        print("Distance:", distance)
                        self.grasp_count = 0
                        self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 4)
                        self.start_time = time.time()
                        print("[OrangeGraspDemo]  Waiting for execution...")
                    continue
                
                self.obj_found_time = time.time()
                    
                target = from_camera_target_to_ee_target(camera_target, arm_pos_reading, self.from_ee_to_cam_trans, self.from_cam_to_ee_target_trans).tolist()

                # (E) Send movement command to the robotic arm
                if (time.time() - self.start_time) > 0.5: # Waiting for arm execution, second 1
                    distance = compute_6d_distance(arm_pos_reading, target)
                    print("Distance:", distance)
                    if distance<= 0.005:
                        self.grasp_count += 1
                        if self.grasp_count >=5:
                            self.write_arm_position.close_fr3_gripper()
                            print("[OrangeGraspDemo] Grasp command sent. Waiting for execution...")
                            time.sleep(1.0)
                            self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 4)
                            self.grasp_count = 0
                            break
                    
                    else:
                        duration = distance/0.06
                        if duration<=1:
                            duration = 1
                        self.grasp_count = 0
                        self.write_arm_position.rt_movec_soft(target, duration)
                        self.start_time = time.time()
                        print("[OrangeGraspDemo]  Waiting for execution...")


        except KeyboardInterrupt:
            print("[OrangeGraspDemo] Ctrl + C received. Exiting...")
        finally:
            self.shutdown()

    def shutdown(self):
        """
        Performs cleanup actions when exiting. If necessary, stops camera/robot arm threads.
        """
        print("[OrangeGraspDemo] Shutting down...")

        # Stop background threads if required
        self.read_arm_position_loop.shutdown()
        self.rs_loop.shutdown()

        print("[OrangeGraspDemo] Shutdown complete.")
        
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
            print("[get_camera_target_from_solo_v2] No objects found by SOLOv2.")
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
