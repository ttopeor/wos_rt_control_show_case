import json
import time
import numpy as np
import matplotlib.pyplot as plt  # <-- 新增引入 matplotlib

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

        self.fr3_home_pos = self.config["fr3_home_pos_stright"]
        fr3_id = self.config["fr3_resource_id"]
        wos_endpoint = self.config["wos_end_point"]
        print("WOS Endpoint:", wos_endpoint)

        self.read_arm_position_loop = ReadArmPositionLoop(fr3_id, wos_endpoint)
        self.read_arm_position_loop.loop_spin(frequency=50)

        self.write_arm_position = robot_rt_control(fr3_id, wos_endpoint)

        # 发送回零位
        self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 3)
        print("Homing Arm fr3...")
        self.waypoints = []

        # 增加一个列表，用于记录所有发送出去的 waypoint
        self.sent_waypoints = []

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

        period = 1.0/20.0
        dtheta = 2.0 * period  # rad
        buffer_len = 5

        try:
            while True:
                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)

                target = [x, y, home_z, roll, pitch, yaw]

                new_wp = {
                    "position": target,
                    "duration": period
                }

                if len(self.waypoints) < buffer_len:
                    self.waypoints.append(new_wp)
                else:
                    # 移除最旧的
                    self.waypoints.pop(0)
                    self.waypoints.append(new_wp)

                    # 发送
                    self.write_arm_position.rt_move(self.waypoints)

                    # 记录“刚刚发送的 waypoint”，这里示例仅把最后一个存下来
                    # 如果想记录一次发送的全部，可以改成 self.sent_waypoints.extend(self.waypoints)
                    self.sent_waypoints.append(new_wp)

                theta += dtheta
                time.sleep(period)

        except KeyboardInterrupt:
            print("Ctrl + C received. Exiting...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down...")
        self.read_arm_position_loop.shutdown()

        # 保存发出的所有 waypoint 到 JSON 文件
        # with open("sent_waypoints.json", "w") as f:
        #     json.dump(self.sent_waypoints, f, indent=2)
        # print("Waypoints saved to sent_waypoints.json")

        # 画出在 XY 平面的轨迹
        # if len(self.sent_waypoints) > 0:
        #     xs = [wp["position"][0] for wp in self.sent_waypoints]
        #     ys = [wp["position"][1] for wp in self.sent_waypoints]

        #     plt.figure()
        #     plt.plot(xs, ys, 'o-', label='Sent Waypoints')
        #     plt.title("Waypoints Trajectory in XY plane")
        #     plt.xlabel("X (m)")
        #     plt.ylabel("Y (m)")
        #     plt.grid(True)
        #     plt.legend()
        #     plt.show()

        print("Shutdown complete.")
        
def main():
    test = RTMoveSTest()
    test.run()

if __name__ == "__main__":
    main()
