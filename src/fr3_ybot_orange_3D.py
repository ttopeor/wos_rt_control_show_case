import os
import time
import cv2
import numpy as np
import copy

from object_segments.detect_obj import ObjDetectorSOLOv2
from utils.config_loader import load_config

from rt_loops.read_rs_frams_loop import ReadRSFrameLoop
from rt_loops.read_arm_pos_loop import ReadArmPositionLoop
from utils.math_tools import average_angle, compute_6d_distance
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

        self.fr3_home_pos = self.config["fr3_home_pos_stright"]
        fr3_id = self.config["fr3_resource_id"]
        wos_endpoint = self.config["wos_end_point"]
        print("[OrangeGraspDemo] WOS Endpoint:", wos_endpoint)

        front_rs_serial = self.config["rs_D405"]["front_camera_serial"]
        back_rs_serial = self.config["rs_D405"]["back_camera_serial"]

        self.from_ee_to_front_cam_trans = self.config["rs_D405"]["mount_transfer_dual_front"]
        self.from_ee_to_back_cam_trans = self.config["rs_D405"]["mount_transfer_dual_back"]

        self.from_front_cam_to_ee_target_trans = self.config["rs_D405"]["target_transfer_dual_front"]
        self.from_back_cam_to_ee_target_trans = self.config["rs_D405"]["target_transfer_dual_back"]


        # 2) Start the RealSense frame acquisition loop
        self.front_rs_loop = ReadRSFrameLoop(freq=30, serial_number=front_rs_serial)
        self.back_rs_loop = ReadRSFrameLoop(freq=30, serial_number=back_rs_serial)

        # 3) Start the robotic arm position reading loop
        self.read_arm_position_loop = ReadArmPositionLoop(fr3_id, wos_endpoint)
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
        self.write_arm_position.rt_move_one_waypoint(self.fr3_home_pos, 3)
        self.write_arm_position.open_fr3_gripper()
        print("[OrangeGraspDemo] Homing Arm fr3...")
        time.sleep(3.5)


        print("[OrangeGraspDemo] Setup completed.")
        self.control_period = 0.6
        self.z_offset = 0.01
        self.last_control_time = time.time()
        
        self.obj_not_found_count = 3
        self.object_stopped_count = 2
        self.if_object_coordinate_init = False
        
    def run(self):
        """
        Continuously executes the object detection, affordance estimation, 
        and robotic grasping process in a loop. 
        Exits on Ctrl+C.
        """
        try:
            while True:                
                # (A) Retrieve the latest camera frames
                color_image_front, depth_image_front = self.front_rs_loop.get_latest_frames()
                color_image_back, depth_image_back = self.back_rs_loop.get_latest_frames()

                if color_image_front is None or depth_image_front is None or color_image_back is None or depth_image_back is None:
                    print("[OrangeGraspDemo] Waiting for camera frames...")
                    time.sleep(0.1)
                    continue

                arm_pos_reading = self.read_arm_position_loop.get_position_reading()

                target = self.get_arm_target_from_dual_cam(arm_pos_reading) 
                
                if target is None:
                    self.obj_not_found_count -=1
                    if self.obj_not_found_count <=0:
                        self.object_stopped_count = 2
                        self.if_object_coordinate_init = False
                        self.last_control_time = time.time()
                        self.obj_not_found_count = 3
                        print("[OrangeGraspDemo] Going home. Looking for object...")
                    continue
                self.obj_not_found_count = 3
                
                target[2] += self.z_offset
                target[3:] = self.fr3_home_pos[3:]
                
                if (time.time() - self.last_control_time) > self.control_period:
    
                    if self.if_object_coordinate_init is False:
                        self.last_obj_coordinate = copy.copy(target)
                        self.if_object_coordinate_init = True
                        self.last_control_time = time.time()
                        continue
                    
                    object_moved_distance = compute_6d_distance(self.last_obj_coordinate, target)
                    
                    if object_moved_distance > 0.005:
                        target_xy = copy.copy(target)
                        target_xy[2] = arm_pos_reading[2]
                        self.write_arm_position.rt_move_one_waypoint(target_xy, 2 * self.control_period)
                        self.last_control_time = time.time()
                        self.object_stopped_count = 2
                        self.last_obj_coordinate = copy.copy(target)
                    #else object not moving count down
                    else:
                        self.object_stopped_count -=1
                        self.last_control_time = time.time()
                        self.last_obj_coordinate = copy.copy(target)
                        
                    # if object not moving and count <= 0, grasp and stop
                    if object_moved_distance < 0.005 and self.object_stopped_count<=0:
                        temp_start_time = time.time()
                        self.write_arm_position.rt_move_one_waypoint(target, 2)
                        print("1:",time.time()-temp_start_time)
                        time.sleep(0.3)
                        print("2:",time.time()-temp_start_time)
                        self.write_arm_position.close_fr3_gripper()
                        print("3:",time.time()-temp_start_time)
                        self.write_arm_position.rt_move_one_waypoint(self.fr3_home_pos, 5)
                        time.sleep(5)
                        break
                


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
        self.front_rs_loop.shutdown()
        self.back_rs_loop.shutdown()

        print("[OrangeGraspDemo] Shutdown complete.")
        
    def get_arm_target_from_dual_cam(self, arm_pos_reading):
        front_cam_target = self.get_camera_target_from_solo_v2(self.front_rs_loop, window_name="FrontCam")
        back_cam_target  = self.get_camera_target_from_solo_v2(self.back_rs_loop,  window_name="BackCam")

        if front_cam_target is None or back_cam_target is None:
            return None

        if front_cam_target is None:
            target = from_camera_target_to_ee_target(back_cam_target, arm_pos_reading, self.from_ee_to_back_cam_trans, self.from_back_cam_to_ee_target_trans).tolist()
            return target
        
        if back_cam_target is None:
            target = from_camera_target_to_ee_target(front_cam_target, arm_pos_reading, self.from_ee_to_front_cam_trans, self.from_front_cam_to_ee_target_trans).tolist()
            return target

        target_by_back_cam = from_camera_target_to_ee_target(back_cam_target, arm_pos_reading, self.from_ee_to_back_cam_trans, self.from_back_cam_to_ee_target_trans).tolist()
        target_by_front_cam = from_camera_target_to_ee_target(front_cam_target, arm_pos_reading, self.from_ee_to_front_cam_trans, self.from_front_cam_to_ee_target_trans).tolist()

        # print("target_by_front_cam:", target_by_front_cam)
        # print("target_by_back_cam:", target_by_back_cam)
        averaged_target = self.average_pose(target_by_front_cam,target_by_back_cam)
        # print("averaged_cam_target:", averaged_target)

        return averaged_target
    
    def average_pose(self, front_pose, back_pose):

        import math
        
        x = (front_pose[0] + back_pose[0]) / 2.0
        y = (front_pose[1] + back_pose[1]) / 2.0
        z = (front_pose[2] + back_pose[2]) / 2.0
        
        roll  = average_angle(front_pose[3], back_pose[3])
        pitch = average_angle(front_pose[4], back_pose[4])
        yaw   = average_angle(front_pose[5], back_pose[5])
    
        return [x, y, z, roll, pitch, yaw]
    
    def get_camera_target_from_solo_v2(self, rs_loop: ReadRSFrameLoop, window_name="SOLOv2 Mask Overlay"):
        color_image, depth_image = rs_loop.get_latest_frames()
        if color_image is None or depth_image is None:
            print("[get_camera_target_from_solo_v2] No camera frames available.")
            return None

        depth_intrin = rs_loop.get_intrinsic()
        depth_scale = rs_loop.get_depth_scale()

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
            cv2.imshow(window_name, vis_img)
            cv2.waitKey(1)
            return None

        cv2.imshow(window_name, vis_img)
        cv2.waitKey(1)

        if results:
            chosen = min(results, key=lambda obj: obj['center_3d'][2])
            (x_c, y_c, z_c) = chosen['center_3d']
            return [-x_c, -y_c, z_c, 0, 0, 0]
        else:
            return None

        
def main():
    demo = OrangeGraspDemo()
    demo.run()

if __name__ == "__main__":
    main()
