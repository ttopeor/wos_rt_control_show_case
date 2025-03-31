#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------- 读取并解析 JSON -------------------
data_path = "/home/yue/Workspace/wos/tmp/logdata.json"
with open(data_path, 'r') as f:
    log_data = json.load(f)

ticks = log_data["ticks"]            # TickLog 数组
set_targets = log_data["set_target"] # SetTargetLog 数组
# ------------------- 解析 ticks (参考/实时状态) -------------------
tick_times = []
tick_joint_pos_ref = []
tick_joint_vel_ref = []
tick_joint_acc_ref = []
tick_joint_pos = []
tick_joint_vel = []
tick_joint_acc = []

for tk in ticks:
    t = tk["time_from_start"]
    tick_times.append(t)

    ref_state = tk["current_joint_state_ref"]
    tick_joint_pos_ref.append(ref_state["position"])
    tick_joint_vel_ref.append(ref_state["velocity"])
    tick_joint_acc_ref.append(ref_state["acceleration"])
    
    cur_state = tk["current_joint_state"]
    tick_joint_pos.append(cur_state["position"])
    tick_joint_vel.append(cur_state["velocity"])
    tick_joint_acc.append(cur_state["acceleration"])

tick_times = np.array(tick_times)
tick_joint_pos_ref = np.array(tick_joint_pos_ref)  # shape: (N, n_joints)
tick_joint_vel_ref = np.array(tick_joint_vel_ref)
tick_joint_acc_ref = np.array(tick_joint_acc_ref)
tick_joint_pos = np.array(tick_joint_pos)
tick_joint_vel = np.array(tick_joint_vel)
tick_joint_acc = np.array(tick_joint_acc)

# ------------------- 解析所有 set_target (各自独立存储) -------------------
plans = []
for st in set_targets:
    st_time = st["time_from_start"]

    # ========== 计算每个 set_target 的 joints_waypoints 的绝对时间 ==========
    wp_times_local = []
    wp_pos_local = []
    cum_time = st_time
    for wp in st["joints_waypoints"]:
        cum_time += wp["duration"]
        wp_times_local.append(cum_time)
        wp_pos_local.append(wp["position"])

    wp_times_local = np.array(wp_times_local)
    wp_pos_local = np.array(wp_pos_local)  # shape: (M, n_joints)

    # ========== 解析 planned_joints_trajectory (如果有的话) ==========
    planned_times_local = []
    planned_pos_local = []
    planned_vel_local = []
    planned_acc_local = []

    if "planned_joints_trajectory" in st:  # 有些情况下可能没有
        for pt in st["planned_joints_trajectory"]:
            t_ns = pt["time_from_start"]
            t_s = t_ns / 1e9
            t_abs = st_time + t_s

            planned_times_local.append(t_abs)

            stt = pt["state"]
            planned_pos_local.append(stt["position"])
            planned_vel_local.append(stt.get("velocity", [0]*6))
            planned_acc_local.append(stt.get("acceleration", [0]*6))

    planned_times_local = np.array(planned_times_local)
    planned_pos_local = np.array(planned_pos_local)
    planned_vel_local = np.array(planned_vel_local)
    planned_acc_local = np.array(planned_acc_local)

    # 存到一个 dict
    plans.append({
        "start_time": st_time,
        "waypoints_times": wp_times_local,
        "waypoints_positions": wp_pos_local,
        "planned_times": planned_times_local,
        "planned_pos": planned_pos_local,
        "planned_vel": planned_vel_local,
        "planned_acc": planned_acc_local
    })

# ------------------- 只可视化第 7 轴 (Python 索引 6) -------------------
joint_idx = 6

# ------------------- 自动计算 y 轴范围 -------------------
all_pos_vals = []
all_vel_vals = []
all_acc_vals = []

if tick_joint_pos_ref.size > 0:
    all_pos_vals.extend(tick_joint_pos_ref[:, joint_idx])
    all_vel_vals.extend(tick_joint_vel_ref[:, joint_idx])
    all_acc_vals.extend(tick_joint_acc_ref[:, joint_idx])
if tick_joint_pos.size > 0:
    all_pos_vals.extend(tick_joint_pos[:, joint_idx])
    all_vel_vals.extend(tick_joint_vel[:, joint_idx])
    all_acc_vals.extend(tick_joint_acc[:, joint_idx])

max_time_val = 0
for p in plans:
    if p["waypoints_positions"].size > 0:
        all_pos_vals.extend(p["waypoints_positions"][:, joint_idx])
        max_time_val = max(max_time_val, p["waypoints_times"].max())

    if p["planned_pos"].size > 0:
        all_pos_vals.extend(p["planned_pos"][:, joint_idx])
        all_vel_vals.extend(p["planned_vel"][:, joint_idx])
        all_acc_vals.extend(p["planned_acc"][:, joint_idx])
        max_time_val = max(max_time_val, p["planned_times"].max())

max_t_tick = tick_times.max() if tick_times.size > 0 else 0
max_time = max(max_t_tick, max_time_val)

if not all_pos_vals:
    all_pos_vals = [0]
if not all_vel_vals:
    all_vel_vals = [0]
if not all_acc_vals:
    all_acc_vals = [0]

pos_min, pos_max = min(all_pos_vals), max(all_pos_vals)
vel_min, vel_max = min(all_vel_vals), max(all_vel_vals)
acc_min, acc_max = min(all_acc_vals), max(all_acc_vals)

def expand_range(vmin, vmax, ratio=0.05):
    if vmin == vmax:
        return vmin - 1, vmax + 1
    span = vmax - vmin
    return vmin - ratio*span, vmax + ratio*span

pos_min, pos_max = expand_range(pos_min, pos_max)
vel_min, vel_max = expand_range(vel_min, vel_max)
acc_min, acc_max = expand_range(acc_min, acc_max)

# ------------------- 准备绘图 -------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].set_ylabel("Position (J7)")
axes[1].set_ylabel("Velocity (J7)")
axes[2].set_ylabel("Acceleration (J7)")
axes[2].set_xlabel("Time (s)")

(line_pos_ref,) = axes[0].plot([], [], 'k.', label='Ref Pos')
(line_vel_ref,) = axes[1].plot([], [], 'k.', label='Ref Vel')
(line_acc_ref,) = axes[2].plot([], [], 'k.', label='Ref Acc')

(line_pos_cur,) = axes[0].plot([], [], 'r^', label='Current Pos')
(line_vel_cur,) = axes[1].plot([], [], 'r^', label='Current Vel')
(line_acc_cur,) = axes[2].plot([], [], 'r^', label='Current Acc')

(line_pos_wp,) = axes[0].plot([], [], 'bo', label='Waypoints', zorder=10)

(line_pos_plan,) = axes[0].plot([], [], 'g.', label='Planned Pos (Future)')
(line_vel_plan,) = axes[1].plot([], [], 'g.', label='Planned Vel (Future)')
(line_acc_plan,) = axes[2].plot([], [], 'g.', label='Planned Acc (Future)')

def init():
    axes[0].set_xlim(0, max_time + 0.1)
    axes[0].set_ylim(pos_min, pos_max)
    axes[1].set_ylim(vel_min, vel_max)
    axes[2].set_ylim(acc_min, acc_max)
    for ax in axes:
        ax.grid(True)
    return (line_pos_ref, line_vel_ref, line_acc_ref,
            line_pos_cur, line_vel_cur, line_acc_cur,
            line_pos_wp,
            line_pos_plan, line_vel_plan, line_acc_plan)

def update(frame_t):
    """frame_t: 当前动画所处的时间 (秒)"""

    # ========== 1) 历史 reference (黑色点) <= frame_t ==========
    if tick_times.size > 0:
        mask_tick = (tick_times <= frame_t)
        ref_t = tick_times[mask_tick]
        ref_pos = tick_joint_pos_ref[mask_tick, joint_idx]
        ref_vel = tick_joint_vel_ref[mask_tick, joint_idx]
        ref_acc = tick_joint_acc_ref[mask_tick, joint_idx]
        line_pos_ref.set_data(ref_t, ref_pos)
        line_vel_ref.set_data(ref_t, ref_vel)
        line_acc_ref.set_data(ref_t, ref_acc)
    else:
        line_pos_ref.set_data([], [])
        line_vel_ref.set_data([], [])
        line_acc_ref.set_data([], [])

    # ========== 2) 当前关节状态 (红三角) - 找最近一个 Tick ==========
    if tick_times.size > 0:
        idx_closest = np.argmin(np.abs(tick_times - frame_t))
        cur_pos_val = tick_joint_pos_ref[idx_closest, joint_idx]
        cur_vel_val = tick_joint_vel_ref[idx_closest, joint_idx]
        cur_acc_val = tick_joint_acc_ref[idx_closest, joint_idx]
    else:
        cur_pos_val, cur_vel_val, cur_acc_val = 0, 0, 0

    line_pos_cur.set_data([frame_t], [cur_pos_val])
    line_vel_cur.set_data([frame_t], [cur_vel_val])
    line_acc_cur.set_data([frame_t], [cur_acc_val])

    # ========== 3) 找到当前激活的 set_target (start_time <= frame_t 的最后一个) ==========
    active_plan = None
    for p in plans:
        if p["start_time"] <= frame_t:
            active_plan = p
        else:
            break

    # 如果没有激活计划，就清空蓝色/绿色点
    if active_plan is None:
        line_pos_wp.set_data([], [])
        line_pos_plan.set_data([], [])
        line_vel_plan.set_data([], [])
        line_acc_plan.set_data([], [])
        # ======== 滚动窗口逻辑 (2秒) ========
        update_xlim(axes, frame_t)
        return (line_pos_ref, line_vel_ref, line_acc_ref,
                line_pos_cur, line_vel_cur, line_acc_cur,
                line_pos_wp,
                line_pos_plan, line_vel_plan, line_acc_plan)

    # ========== 3a) Waypoints (蓝色圆点) —— 显示“<= frame_t”的所有 waypoint ==========
    wp_t_arr = active_plan["waypoints_times"]
    wp_pos_arr = active_plan["waypoints_positions"]  # shape: (M, n_joints)

    if wp_t_arr.size > 0:
        wpt_show_t = wp_t_arr[:]
        wpt_show_p = wp_pos_arr[:, joint_idx]
        line_pos_wp.set_data(wpt_show_t, wpt_show_p)
    else:
        line_pos_wp.set_data([], [])

    # ========== 3b) 规划轨迹 (绿色点) —— 只显示时间 >= frame_t（未来部分） ==========
    plan_t_arr = active_plan["planned_times"]
    plan_p_arr = active_plan["planned_pos"]
    plan_v_arr = active_plan["planned_vel"]
    plan_a_arr = active_plan["planned_acc"]

    if plan_t_arr.size > 0:
        mask_future = (plan_t_arr >= frame_t)
        t_display = plan_t_arr[mask_future]
        p_display = plan_p_arr[mask_future, joint_idx]
        v_display = plan_v_arr[mask_future, joint_idx]
        a_display = plan_a_arr[mask_future, joint_idx]

        line_pos_plan.set_data(t_display, p_display)
        line_vel_plan.set_data(t_display, v_display)
        line_acc_plan.set_data(t_display, a_display)
    else:
        line_pos_plan.set_data([], [])
        line_vel_plan.set_data([], [])
        line_acc_plan.set_data([], [])

    # ========== 最后：更新 x 轴范围，滚动窗口 2s =============
    update_xlim(axes, frame_t)

    return (line_pos_ref, line_vel_ref, line_acc_ref,
            line_pos_cur, line_vel_cur, line_acc_cur,
            line_pos_wp,
            line_pos_plan, line_vel_plan, line_acc_plan)

def update_xlim(axes, frame_t):
    """在动画过程中，将 x 轴限制为 [frame_t - 1, frame_t + 1] 的 2 秒窗口。
       同时做边界检查，避免 < 0 或 > max_time"""
    half_window = 1.0
    left = frame_t - half_window
    right = frame_t + half_window

    # 若超出 [0, max_time]，做相应裁剪
    if left < 0:
        left = 0
        right = 2.0  # 保持窗口宽度 2s
    if right > max_time:
        right = max_time
        left = max_time - 2.0
        if left < 0:
            # 如果 max_time < 2，说明总时长本身就很短
            left = 0
            right = max_time

    for ax in axes:
        ax.set_xlim(left, right)


# 生成动画帧序列
num_frames = 3000
frames = np.linspace(0, max_time, num_frames)

slower_factor = 1.0
animation_interval = int(50 * slower_factor)

ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=False,
                    interval=animation_interval,
                    repeat=False)

axes[0].legend(loc='upper right', fontsize='small')
axes[1].legend(loc='upper right', fontsize='small')
axes[2].legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()
