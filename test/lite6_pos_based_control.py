import math
import time
import numpy as np
import matplotlib.pyplot as plt
from rt_loops.rt_loop import RTLoop
from utils.config_loader import load_config
from utils.math_tools import child_to_parent, parent_to_child, unwrap_angle

from delta6.delta6_analytics import DeltaRobot
from rt_loops.read_encoder_loop import ReadEncoderLoop
from delta6.sensors_calibration import sensor_calibration

from rt_loops.read_arm_pos_loop import ReadArmPositionLoop
from wos_api.robot_rt_control import robot_rt_control

from utils.filter import DataFilter


class MainLoop(RTLoop):
    def __init__(self, freq=50):
        super().__init__(freq=freq)
        self.start_time = None
        self.data_log = []

    def setup(self):
        try:
            config = load_config()
        except FileNotFoundError as e:
            print(e)
            return
        
        lite6_home_pos = config["lite6_home_pos"]
        lite6_id = config["lite6_resource_id"]
        wos_endpoint = config["wos_end_point"]
        delta6_lite6_port_name = config["delta6_lite6_port_name"]
        encoder_dir = config["encoder"]["dir"]

        self.read_arm_position_loop = ReadArmPositionLoop(lite6_id, wos_endpoint)
        self.read_arm_position_loop.loop_spin(frequency=self.freq)
        self.write_arm_position = robot_rt_control(lite6_id, wos_endpoint)

        self.write_arm_position.rt_movec_soft(lite6_home_pos, 5)
        print("Homing Arm Lite6.")
        time.sleep(6)

        sensor_calibration(delta6_lite6_port_name)
        self.read_encoder_loop = ReadEncoderLoop(delta6_lite6_port_name, encoder_dir)
        self.read_encoder_loop.loop_spin(frequency=self.freq)

        time.sleep(1)
        self.delta6_lite6 = DeltaRobot()
        self.target_force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.filter = DataFilter(config, smooth_factor = 0.1, cov_name="delta6_cov") #smaller smoother

        arm_pos_reading = self.read_arm_position_loop.get_position_reading()
        delta6_pose_reading = self.delta6_lite6.get_FK_result()
        target = self.calculate_arm_target_pose(self.target_force, arm_pos_reading, delta6_pose_reading)
        self.last_target = target

        print("Setup completed.")
        time.sleep(1)
        self.start_time = time.time()

        super().setup()

    def loop(self):
        current_time = time.time() - self.start_time

        encoder_reading = self.read_encoder_loop.get_encoder_reading()
        self.delta6_lite6.update(*encoder_reading)
        delta6_end_force = self.delta6_lite6.get_end_force()
        delta6_pose_reading = self.delta6_lite6.get_FK_result()

        arm_pos_reading = self.read_arm_position_loop.get_position_reading()

        target = self.calculate_arm_target_pose(self.target_force, arm_pos_reading, delta6_pose_reading)

        self.data_log.append([current_time, self.target_force.copy(), list(delta6_end_force)])

        target[3] = unwrap_angle(target[3], self.last_target[3])
        target[4] = unwrap_angle(target[4], self.last_target[4])
        target[5] = unwrap_angle(target[5], self.last_target[5])

        self.last_target = target

        # filtered_target = self.filter.filter_data(target)

        # filtered_target = list(map(float, filtered_target))
        # filtered_target = [round(float(x), 5) for x in filtered_target]

        #print("target pose", filtered_target)

        self.write_arm_position.rt_movec(target)

    def shutdown(self):
        super().shutdown()
        print("Shutting down...")

        self.plot_data()

    def calculate_arm_target_pose(self, target_force, arm_pos_reading, delta6_pose_reading):
        current_ee_pose = parent_to_child(arm_pos_reading, delta6_pose_reading)
        delta6_end_pose_target = self.delta6_lite6.calculate_euler_pose(*target_force)

        target_ee_pose = current_ee_pose

        arm_target = child_to_parent(target_ee_pose, delta6_end_pose_target)

        return [float(value) for value in arm_target]
    

    def plot_data(self):
        if not self.data_log:
            print("No data collected.")
            return

        data_log = np.array(self.data_log, dtype=object)

        time_values = data_log[:, 0]
        target_forces = np.array(data_log[:, 1].tolist())  
        current_forces = np.array(data_log[:, 2].tolist())  

        force_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        for i, ax in enumerate(axes.flat):
            ax.plot(time_values, target_forces[:, i], label="Target Force", linestyle="dashed")
            ax.plot(time_values, current_forces[:, i], label="Current Force", linestyle="solid")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Force (N or Nm)")
            ax.set_title(force_labels[i])
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    rt_loop = MainLoop(freq=50)
    rt_loop.setup()
    rt_loop.loop_spin()
