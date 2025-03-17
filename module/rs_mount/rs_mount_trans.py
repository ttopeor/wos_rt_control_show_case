from utils.config_loader import load_config
from utils.math_tools import parent_to_child
try:
    config = load_config()
except FileNotFoundError as e:
    print(e)

camera_trans = config["rs_D405"]["mount_transfer"]

def calculate_camera_pos(arm_ee_pose):
    return parent_to_child(arm_ee_pose,camera_trans)
    
def from_camera_target_to_ee_target(cam_target,arm_ee_pose):
    camera_pos = calculate_camera_pos(arm_ee_pose)
    obj_pos = parent_to_child(camera_pos,cam_target)
    return obj_pos
    