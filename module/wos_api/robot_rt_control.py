import json
import logging
import threading
from wos_api.connection import CreateWSClient


class robot_rt_control:
    def __init__(self, robot_id, wos_endpoint):
        """
        Initialize the ReadPositionLoop by loading configuration and establishing a connection to the robot.
        """
        self.robot_id = robot_id
        self.wos_endpoint = wos_endpoint

        self.client = CreateWSClient(self.wos_endpoint)
        success = self.client.connect()
        if not success:
            logging.error("Connection Failed, quitting.")
            exit(1)  # Exit if connection fails

    def rt_movec_soft(self,target, duration):
            result, err = self.client.run_request(self.robot_id, "rt-move-cartesian-soft", {
                                     "destination": target, "useVelocity": False, "duration": duration, "velocityPercentage": 0, "isRelative": False})
            if err:
                print(f"Robot control, Error occurred: {err}")
            return result
    
    def rt_moves(self,target):
        result, err = self.client.run_request(self.robot_id, "rt-move-sequence", {
                                    "cartesianPosition": target, "scale": 1.0})
        if err:
            print(f"Robot control, Error occurred: {err}")
        return result
        
    def rt_movec(self,target):
        result, err = self.client.run_request(self.robot_id, "rt-move-cartesian", {
                                    "destination": target, "velocityPercentage": 100, "isRelative": False})
        if err:
            print(f"Robot control, Error occurred: {err}")
        return result
    def rt_movec_hard(self,target):
        result, err = self.client.run_request(self.robot_id, "rt-move-cartesian-hard", {
                                    "destination": target, "isRelative": False})
        if err:
            print(f"Robot control, Error occurred: {err}")
        return result
    
    def rt_move(self,waypoints):
        payload = {"waypoints": waypoints}
        # json_str = json.dumps(payload, indent=2)
        # print("DEBUG JSON going to rt_move:\n", json_str)
        
        result, err = self.client.run_request(self.robot_id, "rt-move", payload)
        if err:
            print(f"Robot control, Error occurred: {err}")
        return result
    
    def rt_move_one_waypoint(self,target,duration):
        wp = {
                "position": target,
                "duration": duration
            }
        payload = {"waypoints": [wp]}
        # json_str = json.dumps(payload, indent=2)
        # print("DEBUG JSON going to rt_move:\n", json_str)
        
        result, err = self.client.run_request(self.robot_id, "rt-move", payload)
        if err:
            print(f"Robot control, Error occurred: {err}")
        return result
    
    def gripper_fb(self):
        return
    def open_fr3_gripper(self):
        result, err = self.client.run_action(self.robot_id+"/action", "open-gripper", {"width": 0.08, "speed": 0.05},self.gripper_fb())
        if err:
            print(f"Robot control, Error occurred: {err}")
        return result
    def close_fr3_gripper(self):
        result, err = self.client.run_action(self.robot_id+"/action", "close-gripper", 
                                             {"width": 0.065, "speed": 0.08, "force": 0.1, "epsilon": 0.04}, self.gripper_fb())
        if err:
            print(f"Robot control, Error occurred: {err}")
        return result
    
    
    def open_fr3_gripper_async(self, width=0.08, speed=0.1):

        def _task():
            result, err = self.client.run_action(
                self.robot_id + "/action",
                "open-gripper",
                {"width": width, "speed": speed},
                self.gripper_fb()
            )
            if err:
                print(f"[open_fr3_gripper_async] Error: {err}")

        t = threading.Thread(target=_task, daemon=True)
        t.start()

    def close_fr3_gripper_async(self, width=0.065, speed=0.1, force=0.1, epsilon=0.04):
        def _task():
            result, err = self.client.run_action(
                self.robot_id + "/action",
                "close-gripper",
                {"width": width, "speed": speed, "force": force, "epsilon": epsilon},
                self.gripper_fb()
            )
            if err:
                print(f"[close_fr3_gripper_async] Error: {err}")

        t = threading.Thread(target=_task, daemon=True)
        t.start()