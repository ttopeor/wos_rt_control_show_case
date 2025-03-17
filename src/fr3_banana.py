import os
import time
import numpy as np

from obj_affordance_generator import ObjectAffordanceGenerator
from object_segments.detect_obj import ObjDetectorSOLOv2
from utils.config_loader import load_config

from rt_loops.read_rs_frams_loop import ReadRSFrameLoop
from rt_loops.read_arm_pos_loop import ReadArmPositionLoop
from wos_api.robot_rt_control import robot_rt_control

from rs_mount.rs_mount_trans import from_camera_target_to_ee_target

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(REPO_DIR, 'model')


class BananaGraspDemo:
    """
    A class that demonstrates object detection, affordance-based grasping prediction, 
    and robotic arm control for grasping a banana. The main loop follows a step-by-step 
    process instead of running at a fixed frequency.
    """

    def __init__(self):
        # 1) Load configuration
        try:
            self.config = load_config()
        except FileNotFoundError as e:
            print("[BananaGraspDemo] Configuration file not found:", e)
            raise

        self.fr3_home_pos = self.config["fr3_home_pos"]
        fr3_id = self.config["fr3_resource_id"]
        wos_endpoint = self.config["wos_end_point"]
        print("[BananaGraspDemo] WOS Endpoint:", wos_endpoint)

        # 2) Start the RealSense frame acquisition loop
        self.rs_loop = ReadRSFrameLoop()  # Example: 30 FPS
        # 3) Start the robotic arm position reading loop
        self.read_arm_position_loop = ReadArmPositionLoop(fr3_id, wos_endpoint)
        # Set frequency if the loop has a built-in mechanism
        self.read_arm_position_loop.loop_spin(frequency=50)

        # 4) Initialize robot control interface
        self.write_arm_position = robot_rt_control(fr3_id, wos_endpoint)

        # 5) Move the robotic arm to its home position
        self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 5)
        print("[BananaGraspDemo] Homing Arm fr3...")
        time.sleep(6)

        # 6) Initialize object detection and grasp prediction models
        solov2_config_path = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco.py')
        solov2_ckpt_path  = os.path.join(MODEL_DIR, 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth')
        graspnet_ckpt_path = os.path.join(MODEL_DIR, 'checkpoint-rs.tar')

        self.affordance_generator = ObjectAffordanceGenerator(
            graspnet_checkpoint_path=graspnet_ckpt_path,
            num_point=20000,
            num_view=300,
            collision_thresh=0.01,
            voxel_size=0.01,
        )
        self.obj_detector = ObjDetectorSOLOv2(
            config_file=solov2_config_path,
            checkpoint_file=solov2_ckpt_path,
            target_class='banana',
            score_thr=0.5
        )

        time.sleep(2)  # Allow time for models/threads to initialize
        print("[BananaGraspDemo] Setup completed.")
        self.start_time = None

    def run(self):
        """
        Continuously executes the object detection, affordance estimation, 
        and robotic grasping process in a loop. 
        Exits on Ctrl+C.
        """
        try:
            while True:                
                self.start_time = time.time()

                # (A) Retrieve the latest camera frames
                color_image, depth_image = self.rs_loop.get_latest_frames()
                if color_image is None or depth_image is None:
                    print("[BananaGraspDemo] Waiting for camera frames...")
                    time.sleep(0.1)
                    continue

                # (B) Generate object mask based on detection
                mask = self.prepare_pointcloud_mask(color_image, depth_image)
                if mask is None:
                    continue  # Skip this frame if no object is detected

                # (C) Generate grasp affordance
                camera_target = self.generate_affordance(color_image, depth_image, mask)
                if camera_target is None:
                    continue  # Skip if no valid grasp is found

                # (D) Retrieve robotic arm position and transform target to end-effector coordinates
                arm_pos_reading = self.read_arm_position_loop.get_position_reading()

                target = from_camera_target_to_ee_target(camera_target, arm_pos_reading).tolist()

                # (E) Send movement command to the robotic arm
                self.write_arm_position.rt_movec_soft(target, 10)
                print("[BananaGraspDemo] Grasp command sent. Waiting for execution...")

                # Optional: break or let the robot keep grasping in a loop
                #time.sleep(10.0)  # Prevent overly fast looping
                input("Please enter for next try: ")
                self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 5)
                print("[BananaGraspDemo] Homing Arm fr3...")
                time.sleep(6)

        except KeyboardInterrupt:
            print("[BananaGraspDemo] Ctrl + C received. Exiting...")
        finally:
            self.shutdown()

    def shutdown(self):
        """
        Performs cleanup actions when exiting. If necessary, stops camera/robot arm threads.
        """
        print("[BananaGraspDemo] Shutting down...")

        # Stop background threads if required
        self.read_arm_position_loop.shutdown()
        self.rs_loop.shutdown()

        print("[BananaGraspDemo] Shutdown complete.")

    def prepare_pointcloud_mask(self, color_image, depth_image):
        """
        Detects the object and generates a refined depth mask for grasp planning.
        """
        # 1) Detect objects (e.g., banana)
        masks = self.obj_detector.detect_obj(color_image)
        num_masks = len(masks)
        if num_masks == 0:
            elapsed = time.time() - self.start_time
            print(f"[BananaGraspDemo] No objects found. Time for this frame: {elapsed:.3f} s")
            return None
        else:
            print(f"[BananaGraspDemo] Detected {num_masks} object(s).")

        # 2) Combine all masks
        H, W = depth_image.shape
        mask_bool = np.zeros((H, W), dtype=bool)
        for m in masks:
            mask_bool |= m  # Logical OR operation to merge masks

        # 3) Refine mask using depth distribution
        obj_depths = depth_image[mask_bool & (depth_image > 0)]
        if len(obj_depths) > 0:
            d_mean = np.mean(obj_depths)
            d_std  = np.std(obj_depths)
            lower = d_mean - 2*d_std
            upper = d_mean + 2*d_std
            refined_mask = mask_bool & (depth_image >= lower) & (depth_image <= upper)
        else:
            refined_mask = np.zeros_like(mask_bool, dtype=bool)

        return refined_mask

    def generate_affordance(self, color_image, depth_image, mask):
        """
        Computes grasp affordance from the object mask and depth information.
        """
        # 1) Generate point cloud and features
        end_points, cloud_o3d = self.affordance_generator.process_data(
            color_image=color_image,
            depth_image=depth_image,
            workspace_mask=mask,
            intrinsic=self.rs_loop.get_intrinsic_matrix(),  
            depth_scale=self.rs_loop.get_depth_scale()     
        )
        # 2) Retrieve possible grasp candidates
        gg = self.affordance_generator.get_grasps(end_points)
        gg.nms()
        gg.sort_by_score()

        # 3) No valid grasp found
        if len(gg) == 0:
            print("[BananaGraspDemo] No valid grasps found!")
            return None

        # 4) Perform collision checking
        if self.affordance_generator.collision_thresh > 0:
            points_np = np.asarray(cloud_o3d.points, dtype=np.float32)
            gg = self.affordance_generator.collision_detection(gg, points_np)

        if len(gg) == 0:
            print("[BananaGraspDemo] No valid grasps after collision check!")
            return None

        # 5) Select best grasp
        best_grasp = gg[0]
        if best_grasp.score < 0.5:
            print(f"[BananaGraspDemo] Best grasp score {best_grasp.score:.3f} < 0.7, skipping.")
            return None
        
        # 6) Convert rotation matrix to roll, pitch, yaw
        trans_rot = self.affordance_generator.grasp_trans(best_grasp.translation, best_grasp.rotation_matrix)
        print(f"[BananaGraspDemo] Grasp target is: {trans_rot}, Socre is {best_grasp.score}")
                
        self.affordance_generator.vis_grasps(gg,cloud_o3d,1)
        return trans_rot

def main():
    demo = BananaGraspDemo()
    demo.run()

if __name__ == "__main__":
    main()
