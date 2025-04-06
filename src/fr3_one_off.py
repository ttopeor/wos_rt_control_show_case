import json
import math
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
        time.sleep(4)

    def draw_line(self, target_pos, duration, hz=6.0):
        """
        主函数：用户调用时指定 target_pos (目标末端位置) 和 duration (总时长)。
        内部自动使用 vmax=0.2, acc=0.1 作为上限。
        """
        current_pos = self.read_arm_position_loop.get_position_reading()
        current_pos[3:6] = self.fr3_home_pos[3:6]
        
        waypoints = self._draw_line_with_acc(
            current_pos,
            target_pos,
            duration=duration,
            hz=hz
        )
        return waypoints

    def _draw_line_with_acc(
        self,
        current_pos,
        target_pos,
        duration,
        hz=30.0,
        vmax=0.3,   # 最大线速度(上限)
        acc=0.2     # 最大加速度(上限)
    ):
        """
        使用“梯形速度轨迹”在直线路径上带加减速地插值，且总时长受用户给定的 duration 约束。
        current_pos: [x, y, z, roll, pitch, yaw]
        target_pos:  [x, y, z, roll, pitch, yaw] 或仅 [x, y, z]
        duration:    用户期望的运动总时长 (秒)
        hz:          采样频率 (点/秒)
        vmax, acc:   速度、加速度上限
        返回: waypoints 列表 (每个包含 {"position": [...], "duration": dt})
        """
        # 1) 提取当前/目标 xyz
        current_xyz = np.array(current_pos[0:3])
        
        # 如果 target_pos 只有 3 个分量, 就把姿态保持不变
        if len(target_pos) == 3:
            target_xyz = np.array(target_pos)
            start_ori  = current_pos[3:6]
        else:
            target_xyz = np.array(target_pos[0:3])
            start_ori  = target_pos[3:6]  # 或者保持 current_pos[3:6] 不变, 看你需求

        d = np.linalg.norm(target_xyz - current_xyz)  # 仅算位置距离

        # 若距离很小，直接返回一个近似静止的点
        if d < 1e-6:
            return [{
                "position": list(current_pos),
                "duration": 0.01
            }]

        # -----------------------
        # 2) 先用 vmax、acc 计算“最短可行时间” t_min
        #    (沿用你的梯形速度曲线公式)
        # -----------------------
        # 加速从0到vmax所需时间
        t_acc_ideal = vmax / acc
        # 加速阶段走的距离 d_acc = 0.5 * a * t_acc_ideal^2
        d_acc_ideal = 0.5 * acc * (t_acc_ideal**2)

        # 判断能否跑到 vmax
        if d >= 2 * d_acc_ideal:
            # 能完整经历加速-匀速-减速
            d_const = d - 2 * d_acc_ideal
            t_const = d_const / vmax
            t_min   = t_acc_ideal + t_const + t_acc_ideal
        else:
            # 跑不到 vmax，只有加速和减速 (两段式)
            # 实际能到达的峰值速度
            vm = math.sqrt(d * acc)
            t_acc_ideal = vm / acc
            t_min       = 2 * t_acc_ideal

        # -----------------------
        # 3) 对比 t_min 与用户指定 duration
        # -----------------------
        if t_min > duration:
            # 表示在 acc=0.1, vmax=0.2 的限制下
            # 无法在 duration 内完成
            raise ValueError(
                f"无法在 {duration:.3f} 秒内走完 {d:.3f}m 的距离 "
                f"(已超出最大速度{vmax}和最大加速度{acc}的能力)"
            )
        else:
            # t_min <= duration => 可以放慢速度使得正好走 duration
            total_time = duration

        # 下面要做“插值”时，我们仍然可以先按照“理想梯形曲线”去算位移随时间的函数 s(t),
        # 然后再把所有时间点 (0..t_min) 拉伸到 (0..total_time)。
        # 这样能够保留同样的加、减速形状，只是整体速度变慢。
        # scale_factor = total_time / t_min
        # distance_covered_base(t') 按原先的(0..t_min)时刻
        # 我们新时刻 t = scale_factor * t'
        # => s_new(t) = s_base(t / scale_factor)
        #
        # 下面就复用一段“距离覆盖函数 distance_covered_base(t)”，
        # 然后采样 t in [0..total_time], 映射到 base_time = t / scale_factor.

        scale_factor = total_time / t_min

        # --- 重新定义一个“原始梯形曲线”的距离覆盖函数(基于 t_min 那套) ---
        def distance_covered_base(t):
            # 和最初的逻辑一样，但把 "实际 t_min" 视为最终时刻
            # 需要用到我们上面判断出来的 t_acc_ideal, d_acc_ideal, vm 等
            if d >= 2 * d_acc_ideal:
                # 三段式
                if t <= t_acc_ideal:
                    return 0.5 * acc * (t**2)
                elif t <= (t_acc_ideal + (d - 2 * d_acc_ideal) / vmax):
                    s_before = d_acc_ideal
                    return s_before + vmax * (t - t_acc_ideal)
                else:
                    s_before = d_acc_ideal + (d - 2 * d_acc_ideal - 0.0)
                    # 之所以 -0.0 只是显式看出剩余距离
                    t_dec = t - (t_acc_ideal + (d - 2 * d_acc_ideal) / vmax)
                    # 减速阶段 s_dec = vmax*t_dec - 0.5*a*t_dec^2
                    return s_before + vmax * t_dec - 0.5 * acc * (t_dec**2)
            else:
                # 两段式
                # 实际峰值速度
                vm_local = math.sqrt(d * acc)
                t_acc_local = vm_local / acc
                if t <= t_acc_local:
                    return 0.5 * acc * (t**2)
                else:
                    s_before = 0.5 * acc * (t_acc_local**2)
                    t_dec = t - t_acc_local
                    return s_before + vm_local * t_dec - 0.5 * acc * (t_dec**2)

        # -----------------------
        # 4) 生成离散采样 (waypoints)，并作时间放缩
        # -----------------------
        num_steps = int(total_time * hz)
        if num_steps < 1:
            num_steps = 1

        dt = total_time / num_steps  # 每个插值点之间的时间间隔
        waypoints = []

        for step in range(num_steps + 1):
            t = step * dt
            if t > total_time:
                t = total_time

            # 先映射回原曲线时间
            base_t = t / scale_factor

            s_t = distance_covered_base(base_t)  # 原曲线在 base_t 时刻走的距离
            # 原曲线最大距离就是 d，因此 s_t 不会超过 d
            fraction = s_t / d  # 距离占比(0~1)

            xyz_now = current_xyz + fraction * (target_xyz - current_xyz)
            # 这里只示范保留姿态不变；若需要插值姿态，可在此自行做插值
            pos_now = [
                xyz_now[0], xyz_now[1], xyz_now[2],
                # 如果想保持原姿态不变:
                start_ori[0], start_ori[1], start_ori[2]
            ]
            waypoints.append({
                "position": pos_now,
                "duration": dt  # 每个离散段的期望执行时间
            })

        return waypoints


    def draw_circle(self, radius=0.05, hz=30.0, circle_time=10.0, plane='XY', w_max=1.0, alpha=0.5):
        """
        从当前机械臂位置起，在指定平面画圆，返回带有加减速的离散圆轨迹 waypoints。
        w_max: 最大角速度（弧度/秒）
        alpha: 最大角加速度（弧度/秒²）
        """
        current_pos = self.read_arm_position_loop.get_position_reading()
        x0, y0, z0 = current_pos[0:3]
        roll0, pitch0, yaw0 = self.fr3_home_pos[3:6]
        
        # 计算整圈角度
        total_theta = 2 * np.pi

        # 梯形角速度轨迹
        t_acc = w_max / alpha
        theta_acc = 0.5 * alpha * t_acc**2

        if total_theta >= 2 * theta_acc:
            # 三段式
            theta_const = total_theta - 2 * theta_acc
            t_const = theta_const / w_max
            t_total = 2 * t_acc + t_const
        else:
            # 两段式
            w_peak = np.sqrt(total_theta * alpha)
            t_acc = w_peak / alpha
            t_const = 0
            t_total = 2 * t_acc

        # 若用户传入的 circle_time 不一样，我们对时间轴进行线性缩放
        time_scale = circle_time / t_total
        t_acc *= time_scale
        t_const *= time_scale
        t_total = circle_time

        def angular_distance_covered(t):
            if total_theta >= 2 * theta_acc:
                if t <= t_acc:
                    return 0.5 * alpha * (t / time_scale) ** 2
                elif t <= t_acc + t_const:
                    return theta_acc + w_max * (t - t_acc) / time_scale
                else:
                    t_dec = (t - t_acc - t_const) / time_scale
                    return theta_acc + theta_const + w_max * t_dec - 0.5 * alpha * t_dec**2
            else:
                w_peak = np.sqrt(total_theta * alpha)
                t_peak = w_peak / alpha
                if t <= t_peak * time_scale:
                    return 0.5 * alpha * (t / time_scale) ** 2
                else:
                    t_dec = (t - t_peak * time_scale) / time_scale
                    return 0.5 * w_peak**2 / alpha + w_peak * t_dec - 0.5 * alpha * t_dec**2

        # 生成轨迹点
        num_steps = int(circle_time * hz)
        dt = 1.0 / hz
        theta_list = [angular_distance_covered(i * dt) for i in range(num_steps)]

        # 圆心坐标
        if plane == 'XY':
            cx, cy = x0 + radius, y0
        else:
            raise NotImplementedError(f"Plane '{plane}' not implemented.")

        waypoints = []

        for theta in theta_list:
            x = cx - radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = z0
            pos = [x, y, z, roll0, pitch0, yaw0]
            waypoints.append({
                "position": pos,
                "duration": dt
            })

        return waypoints

    def run(self):
        """
        演示：先画一条带加减速的直线，再可选画圆。
        """
        table_height = 0.2588
        x_center = 0.4
        
        line_length = 0.2
        circle_radius = 0.05
        
        print("Start drawing line demo ...")
        target_pos = [x_center, -line_length/2, table_height]
        line_waypoints = self.draw_line(target_pos, duration=3.0, hz=2)
        print("Send line_waypoints to robot ...")
        self.write_arm_position.rt_move(line_waypoints)
        time.sleep(3.5)
        
        target_pos = [x_center, line_length/2, table_height]
        line_waypoints = self.draw_line(target_pos, duration=3.0, hz=2)
        print("Send line_waypoints to robot ...")
        self.write_arm_position.rt_move(line_waypoints)
        time.sleep(3.5)
        

        target_pos = [x_center, 0.0, table_height+0.03]
        line_waypoints = self.draw_line(target_pos, duration=2.0, hz=2)
        print("Send line_waypoints to robot ...")
        self.write_arm_position.rt_move(line_waypoints)
        time.sleep(2.5)

        target_pos = [x_center, 0.0, table_height]
        line_waypoints = self.draw_line(target_pos, duration=1.0, hz=2)
        print("Send line_waypoints to robot ...")
        self.write_arm_position.rt_move(line_waypoints)
        time.sleep(1.5)
        
        print("Start drawing circle demo ...")
        circle_waypoints = self.draw_circle(radius=circle_radius, hz=3, circle_time=6.0, plane='XY')
        circle_waypoints.append(
                {
                    "position": [x_center, 0.0, table_height+0.03, self.fr3_home_pos[3],self.fr3_home_pos[4], self.fr3_home_pos[5]],
                    "duration": 1.0
                })
        
        self.write_arm_position.rt_move(circle_waypoints)
        time.sleep(4.5)

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
