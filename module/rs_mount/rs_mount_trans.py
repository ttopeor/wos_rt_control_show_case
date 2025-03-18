from utils.config_loader import load_config
from utils.math_tools import parent_to_child
try:
    config = load_config()
except FileNotFoundError as e:
    print(e)

    
def from_camera_target_to_ee_target(cam_target_local,arm_ee_pose_global,from_ee_to_cam_trans, from_cam_to_ee_target_trans):

    camera_pose_global = parent_to_child(arm_ee_pose_global,from_ee_to_cam_trans)
    #print("camera_pose_global:", camera_pose_global)
    #print("cam_target_local:", cam_target_local)
    ee_target_local = parent_to_child(cam_target_local, from_cam_to_ee_target_trans)
    #print("ee_target_local:", ee_target_local)
    
    ee_target_global = parent_to_child(camera_pose_global,ee_target_local)
    #print("ee_target_global:", ee_target_global)

    return ee_target_global
    
    
    
