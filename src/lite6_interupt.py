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

        self.lite6_home_pos = self.config["lite6_pos_2"]
        lite6_id = self.config["lite6_coodinate_resource_id"]
        wos_endpoint = self.config["wos_end_point"]
        print("WOS Endpoint:", wos_endpoint)

        self.write_arm_position = robot_rt_control(lite6_id, wos_endpoint)
        self.write_arm_position.rt_movec_soft(self.lite6_home_pos, 5)
        print("Homing Arm lite6...")
        self.waypoints = []
        time.sleep(5.5)

    def run(self):
        start_pos = self.lite6_home_pos
        end_pos = [start_pos[0], 0.20, 0.10, start_pos[3], start_pos[4], start_pos[5]]
        end_pos2 = [start_pos[0], -0.20, 0.10, start_pos[3], start_pos[4], start_pos[5]]

        waypoints = []
        period = 5.0  
        num_steps = 5

        for i in range(1, num_steps + 1):
            t = i / num_steps
            alpha = 3 * t**2 - 2 * t**3  # cubic easing

            interpolated_y = (1 - alpha) * start_pos[1] + alpha * end_pos[1]
            interpolated_z = (1 - alpha) * start_pos[2] + alpha * end_pos[2]

            interpolated_pos = [
                start_pos[0],
                interpolated_y,
                interpolated_z,
                start_pos[3],
                start_pos[4],
                start_pos[5]
            ]

            new_wp = {
                "position": interpolated_pos,
                "duration": period / num_steps
            }
            waypoints.append(new_wp)
            
        #interrupted
        interrupted_waypoints = []
        interrupted_num_steps = 2
        
        start_pos = waypoints[2]["position"]
        end_pos = end_pos2
        
        alpha = [0.8, 1.0]
        durations = [3.0,1.0]
        
        for i in range(1, interrupted_num_steps + 1):
            interpolated_y = (1 - alpha[i-1]) * start_pos[1] + alpha[i-1] * end_pos[1]
            interpolated_z = (1 - alpha[i-1]) * start_pos[2] + alpha[i-1] * end_pos[2]

            interpolated_pos = [
                start_pos[0],
                interpolated_y,
                interpolated_z,
                start_pos[3],
                start_pos[4],
                start_pos[5]
            ]

            new_wp = {
                "position": interpolated_pos,
                "duration": durations[i-1]
            }
            interrupted_waypoints.append(new_wp)
            
        print("\nMain Waypoints:")
        print(json.dumps(waypoints, indent=2))
        print("\nInterrupted Waypoints:")
        print(json.dumps(interrupted_waypoints, indent=2))
        
        self.write_arm_position.rt_move(waypoints)
        
        time.sleep(period * 2/5)
        
        self.write_arm_position.rt_move(interrupted_waypoints)


    def shutdown(self):
        print("Shutting down...")
        print("Shutdown complete.")
        
def main():
    test = RTMoveSTest()
    test.run()

if __name__ == "__main__":
    main()
