import json
import time
import numpy as np

from utils.config_loader import load_config

from rt_loops.read_arm_pos_loop import ReadArmPositionLoop
from wos_api.robot_rt_control import robot_rt_control


class RTMoveSTest:
    def __init__(self):
        try:
            self.config = load_config()
        except FileNotFoundError as e:
            print("[OrangeGraspDemo] Configuration file not found:", e)
            raise

        self.fr3_home_pos = self.config["fr3_home_pos"]
        fr3_id = self.config["fr3_resource_id"]
        wos_endpoint = self.config["wos_end_point"]
        print("WOS Endpoint:", wos_endpoint)

        self.read_arm_position_loop = ReadArmPositionLoop(fr3_id, wos_endpoint)
        self.read_arm_position_loop.loop_spin(frequency=50)

        self.write_arm_position = robot_rt_control(fr3_id, wos_endpoint)

        self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 3)
        print("Homing Arm fr3...")
        self.waypoints = []
        time.sleep(3)

    def run(self):
        radius = 0.04 

        home_x = self.fr3_home_pos[0]
        home_y = self.fr3_home_pos[1]
        home_z = self.fr3_home_pos[2]
        roll   = self.fr3_home_pos[3]
        pitch  = self.fr3_home_pos[4]
        yaw    = self.fr3_home_pos[5]

        center_x = home_x - radius
        center_y = home_y

        theta = 0.0
        dtheta = 0.3  # rad

        try:
            while True:
                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)

                target = [x, y, home_z, roll, pitch, yaw]

                new_wp = {
                    "position": target,
                    "duration": 0.2
                }

                if len(self.waypoints) < 5:
                    self.waypoints.append(new_wp)
                else:
                    # Maintain fixed length = 5
                    self.waypoints.pop(0)  # remove oldest
                    self.waypoints.append(new_wp)
                    
                    self.write_arm_position.rt_move(self.waypoints)

                theta += dtheta
                time.sleep(0.2)

        except KeyboardInterrupt:
            print("Ctrl + C received. Exiting...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down...")
        self.read_arm_position_loop.shutdown()
        print("Shutdown complete.")
        
def main():
    test = RTMoveSTest()
    test.run()

if __name__ == "__main__":
    main()
