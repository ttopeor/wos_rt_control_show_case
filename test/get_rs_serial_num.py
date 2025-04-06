import pyrealsense2 as rs

context = rs.context()
connected_devices = []
for d in context.query_devices():
    device_info = {}
    device_info["name"] = d.get_info(rs.camera_info.name)
    device_info["serial"] = d.get_info(rs.camera_info.serial_number)
    connected_devices.append(device_info)

print("Found devices:")
for dev in connected_devices:
    print(f"  Device Name: {dev['name']}, S/N: {dev['serial']}")