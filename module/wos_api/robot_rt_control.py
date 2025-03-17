import logging
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