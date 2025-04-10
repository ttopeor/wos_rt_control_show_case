import time

import numpy as np
from rt_loops.rt_loop import RTLoop
from utils.admitance_controller_6d import AdmittanceController6D
from utils.config_loader import load_config
from utils.PID_controller_6d import PIDController6D
from utils.math_tools import calculate_delta_position, child_to_parent, parent_to_child, quantize_to_resolution

from delta6.delta6_analytics import DeltaRobot
from rt_loops.read_encoder_loop import ReadEncoderLoop
from delta6.sensors_calibration import sensor_calibration

from rt_loops.read_arm_pos_loop import ReadArmPositionLoop
from wos_api.robot_rt_control import robot_rt_control

import pygame
import os
import time
import threading

class MainLoop(RTLoop):
    def __init__(self, freq=50):
        super().__init__(freq=freq)
        self._stop_keyboard_listen = threading.Event()  

    def setup(self):
        # Attempt to load configuration
        try:
            config = load_config()
        except FileNotFoundError as e:
            print(e)
            return
        
        self._keyboard_thread = threading.Thread(
            target=self._listen_keyboard, 
            daemon=True
        )
        self._keyboard_thread.start()
        
        
        lite6_home_pos = config["lite6_home_pos"]
        lite6_id = config["lite6_resource_id"]
        fr3_home_pos = config["fr3_home_pos_stright"]
        fr3_id = config["fr3_resource_id"]
        wos_endpoint = config["wos_end_point"]
        delta6_lite6_port_name =config["delta6_lite6_port_name"]
        encoder_dir = config["encoder"]["dir"]

        adm_config = config["admittance_control_stable"]
        
        M = adm_config["M"] 
        B = adm_config["B"]  
        K = adm_config["K"] 
        
        self.admittance6d = AdmittanceController6D(M, B, K)
        
        self.read_lite6_position_loop = ReadArmPositionLoop(lite6_id, wos_endpoint)
        self.read_lite6_position_loop.loop_spin(frequency=self.freq)

        self.write_lite6_position = robot_rt_control(lite6_id, wos_endpoint)
        self.write_fr3_position = robot_rt_control(fr3_id, wos_endpoint)

        self.write_lite6_position.rt_movec_soft(lite6_home_pos, 5)
        self.write_fr3_position.rt_movec_soft(fr3_home_pos, 5)
        self.write_fr3_position.open_fr3_gripper_async()
        print("Homing Arms")
        time.sleep(6)


        sensor_calibration(delta6_lite6_port_name)
        self.read_encoder_loop = ReadEncoderLoop(delta6_lite6_port_name,encoder_dir)
        self.read_encoder_loop.loop_spin(frequency=self.freq)

        time.sleep(1)
        self.delta6_lite6 = DeltaRobot()
        self.target_force = [0.0,0.0,0.0,0.0,0.0,0.0]

        self.fr3_ee_trans = [0,0,0.134,0,0,0]
        self.lite6_ee_home_pos = parent_to_child(lite6_home_pos,self.delta6_lite6.forward_kinematics(*self.read_encoder_loop.get_encoder_reading()))
        self.fr3_ee_home_pos = parent_to_child(fr3_home_pos,self.fr3_ee_trans)

        self.last_fr3_control_time = time.time()
        self.fr3_counter = 0
        self.fr3_target_buffer = []
        self.fr3_buffer_len = 5
        print("Setup completed.")
        time.sleep(1)

        super().setup()

    def loop(self):
        encoder_reading = self.read_encoder_loop.get_encoder_reading()
        self.delta6_lite6.update(*encoder_reading)
        delta6_pose_reading = self.delta6_lite6.get_FK_result()
        delta6_end_force = self.delta6_lite6.get_end_force()

        lite6_pos_reading = self.read_lite6_position_loop.get_position_reading()

        lite6_target = self.admt_controller( 
            delta6_end_force,    
            lite6_pos_reading,
            delta6_pose_reading
        )
                
        self.write_lite6_position.rt_movec(lite6_target)
        
        self.fr3_counter += 1
        if self.fr3_counter >= 2:
            self.fr3_counter = 0
            fr3_target = self.pos_controller(lite6_pos_reading, delta6_pose_reading)
            #print("fr3_target:", [float(round(value, 3)) for value in fr3_target])
            fr3_target_wp = {
                "position": fr3_target,
                "duration": time.time() - self.last_fr3_control_time
            }
            self.last_fr3_control_time = time.time()

            if len(self.fr3_target_buffer) < self.fr3_buffer_len:
                    self.fr3_target_buffer.append(fr3_target_wp)
            else:
                self.fr3_target_buffer.pop(0)
                self.fr3_target_buffer.append(fr3_target_wp)
                self.write_fr3_position.rt_move(self.fr3_target_buffer)
                #print(self.fr3_target_buffer)
        
    def shutdown(self):
        self._stop_keyboard_listen.set()
        super().shutdown()
        print("MainLoop shutdown complete.")
    
    def pos_controller(self, lite6_pos_reading, delta6_pose_reading):
        current_lite6_ee_pose = parent_to_child(lite6_pos_reading, delta6_pose_reading)

        lite6_ee_pose_diff = calculate_delta_position(self.lite6_ee_home_pos, current_lite6_ee_pose)
        fr3_ee_pose_diff = lite6_ee_pose_diff

        target_fr3_ee_pose = parent_to_child(self.fr3_ee_home_pos, fr3_ee_pose_diff)
        target_fr3_pose = child_to_parent(target_fr3_ee_pose, self.fr3_ee_trans)
        target_fr3_pose = [float(value) for value in target_fr3_pose]
        return target_fr3_pose
    
    def admt_controller(self, current_force, arm_pos_reading, delta6_pose_reading):
        current_ee_pose = parent_to_child(arm_pos_reading, delta6_pose_reading)

        force_error = [-current_force[i] for i in range(6)]
        #force_error = [force_error[0],force_error[1],force_error[2],0,0,0]
        dt = 1.0 / self.freq
        
        delta6_end_pose_diff = self.admittance6d.update(force_error, dt)

        delta6_end_pose_target = np.array(delta6_pose_reading) + np.array(delta6_end_pose_diff)

        target_ee_pose = current_ee_pose

        arm_target = child_to_parent(target_ee_pose, delta6_end_pose_target)

        return [float(value) for value in arm_target]
    
    def _listen_keyboard(self):
        
        pygame.init()
        screen = pygame.display.set_mode((480, 480))
        pygame.display.set_caption("Keyboard Listener")

        print("Keyboard listener (pygame) started. Press w=OpenGripper, s=CloseGripper, q=Quit")

        clock = pygame.time.Clock()

        while not self._stop_keyboard_listen.is_set():
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        print(">>> [Keyboard] Open Gripper command")
                        self.write_fr3_position.open_fr3_gripper_async()
                    elif event.key == pygame.K_s:
                        print(">>> [Keyboard] Close Gripper command")
                        self.write_fr3_position.close_fr3_gripper_async()
                    elif event.key == pygame.K_q:
                        print(">>> [Keyboard] Quit request received.")
                        self.shutdown()
                        break

            clock.tick(30)

        pygame.quit()
        print("Keyboard raw-mode listener exit.")
    

if __name__ == "__main__":
    rt_loop = MainLoop(freq=50)
    rt_loop.setup()
    rt_loop.loop_spin()
