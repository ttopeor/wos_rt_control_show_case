import os
import time

from utils.config_loader import load_config
from wos_api.robot_rt_control import robot_rt_control

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(REPO_DIR, 'model')


class GraspTest:

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

        self.write_arm_position = robot_rt_control(fr3_id, wos_endpoint)

        time.sleep(2)  # Allow time for models/threads to initialize
        
        # self.write_arm_position.open_fr3_gripper()
        # time.sleep(3.5)

        
    def run(self):

        try:
            temp_start_time = time.time()

            print("1:",time.time()-temp_start_time)
            self.write_arm_position.open_fr3_gripper_async()
            print("2:",time.time()-temp_start_time)

            time.sleep(5)
            
            print("3:",time.time()-temp_start_time)
            self.write_arm_position.close_fr3_gripper_async()
            print("4:",time.time()-temp_start_time)
            
        except KeyboardInterrupt:
            print("[OrangeGraspDemo] Ctrl + C received. Exiting...")
        finally:
            self.shutdown()

    def shutdown(self):
        """
        Performs cleanup actions when exiting. If necessary, stops camera/robot arm threads.
        """
        print("Shutting down...")
        
def main():
    demo = GraspTest()
    demo.run()

if __name__ == "__main__":
    main()
