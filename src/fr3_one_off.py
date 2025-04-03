import json
import time
import numpy as np

from utils.config_loader import load_config
from rt_loops.read_arm_pos_loop import ReadArmPositionLoop
from wos_api.robot_rt_control import robot_rt_control


class RTMoveSOneOffDemo:
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

        # 读取臂位姿循环 (后台线程)
        self.read_arm_position_loop = ReadArmPositionLoop(fr3_id, wos_endpoint)
        self.read_arm_position_loop.loop_spin(frequency=100)

        # 写指令接口
        self.write_arm_position = robot_rt_control(fr3_id, wos_endpoint)

        # 先回到 home
        self.write_arm_position.rt_movec_soft(self.fr3_home_pos, 3)
        print("Homing Arm fr3...")
        time.sleep(3)

    def draw_line(self, target_pos, hz=30.0, vmax=0.2, acc=0.1):
        """
        带加减速的直线插值生成 waypoints。
        :param target_pos: 目标末端位置 [x, y, z, roll, pitch, yaw]
        :param hz:   采样频率 (点/秒)
        :param vmax: 最大线速度 (米/秒)
        :param acc:  最大加速度 (米/秒^2)
        :return:     waypoints 列表
        """
        current_pos = self.read_arm_position_loop.get_position_reading()
        waypoints = self._draw_line_with_acc(
            current_pos, target_pos, hz, vmax, acc
        )
        return waypoints

    def _draw_line_with_acc(self, current_pos, target_pos, hz=30.0, vmax=0.2, acc=0.1):
        """
        使用“梯形速度轨迹”在直线路径上带加减速地插值。
        current_pos: [x, y, z, roll, pitch, yaw]
        target_pos:  [x, y, z, roll, pitch, yaw]
        返回: waypoints 列表 (每个包含 "position" 和 "duration")
        """
        # -----------------------
        # 1) 计算3D路径总距离 d
        # -----------------------
        current_xyz = np.array(current_pos[0:3])
        target_xyz  = np.array(target_pos[0:3])
        d = np.linalg.norm(target_xyz - current_xyz)  # 仅算位置距离

        # 若距离很小，直接返回一个近似静止的点
        if d < 1e-6:
            return [{
                "position": current_pos,
                "duration": 0.01
            }]

        # -----------------------
        # 2) 计算梯形速度曲线加减速时间、距离
        # -----------------------
        # 加速从0到vmax所需时间
        t_acc = vmax / acc
        # 加速阶段走的距离 d_acc = 0.5 * a * t_acc^2
        d_acc = 0.5 * acc * t_acc**2

        # 判断能否到达 vmax
        if d >= 2 * d_acc:
            # 能完整经历加速-匀速-减速
            d_const = d - 2 * d_acc   # 匀速段距离
            t_const = d_const / vmax  # 匀速段时间
            total_time = t_acc + t_const + t_acc
        else:
            # 跑不到 vmax，只有加速和减速 (两段式)
            vm = np.sqrt(d * acc)     # 实际能达到的峰值速度
            t_acc = vm / acc
            t_const = 0.0
            total_time = 2 * t_acc

        # -----------------------
        # 3) 生成离散采样 (waypoints)
        # -----------------------
        num_steps = int(total_time * hz)
        if num_steps < 1:
            num_steps = 1

        dt = 1.0 / hz
        waypoints = []

        # 定义一个函数，用来在任意时刻 t 计算“已行驶距离” s(t)
        def distance_covered(t):
            # 三段式情况
            if d >= 2 * d_acc:
                # A. 加速段
                if t <= t_acc:
                    return 0.5 * acc * (t**2)
                # B. 匀速段
                elif t <= (t_acc + t_const):
                    s_before = d_acc
                    return s_before + vmax * (t - t_acc)
                # C. 减速段
                else:
                    s_before = d_acc + (vmax * t_const)  # 加速+匀速
                    t_dec = t - (t_acc + t_const)
                    # 减速阶段 s_dec = vmax*t_dec - 0.5*a*t_dec^2
                    return s_before + vmax * t_dec - 0.5 * acc * (t_dec**2)
            else:
                # 两段式 (加速-减速)
                if t <= t_acc:
                    # 加速阶段
                    return 0.5 * acc * (t**2)
                else:
                    # 减速阶段
                    s_before = 0.5 * acc * (t_acc**2)
                    t_dec = t - t_acc
                    # vm = a * t_acc
                    return s_before + (acc * t_acc) * t_dec - 0.5 * acc * (t_dec**2)

        # 姿态简单线性插值
        start_ori = np.array(current_pos[3:6])
        end_ori   = np.array(target_pos[3:6])
        delta_ori = end_ori - start_ori

        for step in range(num_steps + 1):
            t = step * dt
            if t > total_time:
                t = total_time

            s_t = distance_covered(t)     # 当前已走距离
            fraction = s_t / d            # 距离占比 (0~1)
            # 位置插值
            xyz_now = current_xyz + fraction * (target_xyz - current_xyz)
            # 姿态插值
            ori_now = start_ori + fraction * delta_ori

            pos_now = [
                xyz_now[0], xyz_now[1], xyz_now[2],
                ori_now[0], ori_now[1], ori_now[2]
            ]
            waypoints.append({
                "position": pos_now,
                "duration": dt  # 每个离散段的期望时间
            })

        return waypoints

    def draw_circle(self, radius=0.05, hz=30.0, circle_time=10.0, plane='XY'):
        """
        从当前机械臂位置起，在指定平面画圆，返回离散圆轨迹的 waypoints。
        plane='XY' 仅作示例。
        """
        current_pos = self.read_arm_position_loop.get_position_reading()
        x0, y0, z0, roll0, pitch0, yaw0 = current_pos

        num_steps = int(circle_time * hz)
        if num_steps < 1:
            num_steps = 1

        dt = 1.0 / hz

        if plane == 'XY':
            cx, cy = x0, y0 + radius
            start_theta = 0.0
        else:
            raise NotImplementedError(f"Plane '{plane}' not implemented.")

        waypoints = []
        for step in range(num_steps + 1):
            # 画一整圈
            theta = start_theta + 2.0 * np.pi * (step + 1) / num_steps
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = z0
            roll, pitch, yaw = roll0, pitch0, yaw0
            pos = [x, y, z, roll, pitch, yaw]

            waypoints.append({
                "position": pos,
                "duration": dt
            })

        return waypoints

    def run(self):
        """
        演示：先画一条带加减速的直线，再可选画圆。
        """
        print("Start drawing line demo ...")
        target_pos = [0.4, 0.0, 0.33, -3.14, 0.0, 1.57]

        # 生成带加减速的直线轨迹
        line_waypoints = self.draw_line(
            target_pos=target_pos,
            hz=30.0,   # 插值频率
            vmax=0.2,  # 最大线速度
            acc=0.1    # 最大加速度
        )
        print("Send line_waypoints to robot ...")
        self.write_arm_position.rt_move(line_waypoints)

        # 等待执行完毕 (此处简单 sleep，具体可根据 total_time 计算或做更复杂的判断)
        time.sleep(5.0)

        # 如果需要画圆，可取消注释以下代码
        # print("Start drawing circle demo ...")
        # circle_waypoints = self.draw_circle(radius=0.05, hz=30, circle_time=5.0, plane='XY')
        # self.write_arm_position.rt_move(circle_waypoints)
        # time.sleep(6.0)

        print("Demo finished.")

    def shutdown(self):
        print("Shutting down...")
        self.read_arm_position_loop.shutdown()
        print("Shutdown complete.")


def main():
    test = RTMoveSOneOffDemo()
    try:
        test.run()
    except KeyboardInterrupt:
        print("Ctrl + C received. Exiting...")
    finally:
        test.shutdown()


if __name__ == "__main__":
    main()
