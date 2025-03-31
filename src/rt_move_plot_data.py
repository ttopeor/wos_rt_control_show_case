#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage:

    animator = JointTrajectoryAnimator(
        data_path="/home/yue/Workspace/wos/tmp/logdata.json",
        joint_idx=6,         # Visualize joint #7 (Python index=6)
        half_window=0.4,     # Window half-size for rolling display
        slower_factor=1.0,   # Slow down factor for animation
    )
    animator.run()
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class JointTrajectoryAnimator:
    """
    A class to animate a specific joint (e.g. joint #7) from a given JSON log
    file containing 'ticks' and 'set_target' data.

    The animation will display:
      - Reference trajectory (black dots)
      - Current state (red triangle)
      - Waypoints (blue dots)
      - Planned future trajectory (green dots)
      - With a rolling time window to keep the current time in the middle.

    Parameters
    ----------
    data_path : str
        Path to the JSON log file.
    joint_idx : int
        Which joint index to visualize (Python-based index).
    half_window : float
        Half of the rolling window width in seconds, used to set x-axis range
        around the current time.
    slower_factor : float
        A multiplier to slow down (or speed up) the animation frame interval.
    """

    def __init__(self, 
                 data_path: str,
                 joint_idx: int = 6,
                 half_window: float = 0.4,
                 slower_factor: float = 1.0):
        """
        Constructor. Reads and parses the JSON log. Initializes matplotlib
        figure and lines, but does not start the animation yet.
        """
        self.data_path = data_path
        self.joint_idx = joint_idx
        self.half_window = half_window
        self.slower_factor = slower_factor

        # 1) Load JSON
        with open(self.data_path, 'r') as f:
            log_data = json.load(f)

        # Retrieve tick logs
        self.ticks = log_data.get("ticks", [])
        # Retrieve set_target logs
        self.set_targets = log_data.get("set_target", [])

        # 2) Parse tick data
        self._parse_tick_data()

        # 3) Parse set_target data
        self._parse_set_target_data()

        # 4) Determine overall max_time and Y-limits
        self._compute_plot_ranges()

        # 5) Prepare matplotlib figure and lines
        self._init_figure()

        # 6) Prepare frames for animation
        self._prepare_animation_frames()

        # The actual animation object (FuncAnimation) will be created in run()
        self.ani = None

    def _parse_tick_data(self):
        """
        Parse the 'ticks' array from the log. Extract time, reference states,
        and current states for the selected joint.
        """
        self.tick_times = []
        self.tick_joint_pos_ref = []
        self.tick_joint_vel_ref = []
        self.tick_joint_acc_ref = []

        self.tick_joint_pos = []
        self.tick_joint_vel = []
        self.tick_joint_acc = []

        for tk in self.ticks:
            t = tk["time_from_start"]
            self.tick_times.append(t)

            ref_state = tk["current_joint_state_ref"]
            self.tick_joint_pos_ref.append(ref_state["position"])
            self.tick_joint_vel_ref.append(ref_state["velocity"])
            self.tick_joint_acc_ref.append(ref_state["acceleration"])

            cur_state = tk["current_joint_state"]
            self.tick_joint_pos.append(cur_state["position"])
            self.tick_joint_vel.append(cur_state["velocity"])
            self.tick_joint_acc.append(cur_state["acceleration"])

        self.tick_times = np.array(self.tick_times)
        self.tick_joint_pos_ref = np.array(self.tick_joint_pos_ref)    # shape: (N, n_joints)
        self.tick_joint_vel_ref = np.array(self.tick_joint_vel_ref)
        self.tick_joint_acc_ref = np.array(self.tick_joint_acc_ref)
        self.tick_joint_pos = np.array(self.tick_joint_pos)
        self.tick_joint_vel = np.array(self.tick_joint_vel)
        self.tick_joint_acc = np.array(self.tick_joint_acc)

    def _parse_set_target_data(self):
        """
        Parse the 'set_target' array to extract each target's:
            - start_time
            - waypoints (time + position)
            - planned trajectory (time + pos/vel/acc)
        Store them in a list of dict named 'plans'.
        """
        self.plans = []
        for st in self.set_targets:
            st_time = st["time_from_start"]
            wp_times_local = []
            wp_pos_local = []

            # Accumulate durations to get absolute time for waypoints
            cum_time = st_time
            for wp in st["joints_waypoints"]:
                cum_time += wp["duration"]
                wp_times_local.append(cum_time)
                wp_pos_local.append(wp["position"])

            wp_times_local = np.array(wp_times_local)
            wp_pos_local = np.array(wp_pos_local)  # shape: (M, n_joints)

            # planned trajectory
            planned_times_local = []
            planned_pos_local = []
            planned_vel_local = []
            planned_acc_local = []

            if "planned_joints_trajectory" in st:
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

            self.plans.append({
                "start_time": st_time,
                "waypoints_times": wp_times_local,
                "waypoints_positions": wp_pos_local,
                "planned_times": planned_times_local,
                "planned_pos": planned_pos_local,
                "planned_vel": planned_vel_local,
                "planned_acc": planned_acc_local
            })

    def _compute_plot_ranges(self):
        """
        Determine the maximum time for the entire animation, and the global
        min/max for position, velocity, and acceleration for the selected joint.
        """
        self.all_pos_vals = []
        self.all_vel_vals = []
        self.all_acc_vals = []

        # Tick ref
        if self.tick_joint_pos_ref.size > 0:
            self.all_pos_vals.extend(self.tick_joint_pos_ref[:, self.joint_idx])
            self.all_vel_vals.extend(self.tick_joint_vel_ref[:, self.joint_idx])
            self.all_acc_vals.extend(self.tick_joint_acc_ref[:, self.joint_idx])

        # Tick actual
        if self.tick_joint_pos.size > 0:
            self.all_pos_vals.extend(self.tick_joint_pos[:, self.joint_idx])
            self.all_vel_vals.extend(self.tick_joint_vel[:, self.joint_idx])
            self.all_acc_vals.extend(self.tick_joint_acc[:, self.joint_idx])

        self.max_time_val = 0

        # Parse set_target
        for plan in self.plans:
            wp_pos_arr = plan["waypoints_positions"]
            wp_t_arr   = plan["waypoints_times"]
            if wp_pos_arr.size > 0:
                self.all_pos_vals.extend(wp_pos_arr[:, self.joint_idx])
                local_max = wp_t_arr.max()
                self.max_time_val = max(self.max_time_val, local_max)

            p_pos_arr = plan["planned_pos"]
            p_t_arr   = plan["planned_times"]
            if p_pos_arr.size > 0:
                self.all_pos_vals.extend(p_pos_arr[:, self.joint_idx])
                p_vel_arr = plan["planned_vel"][:, self.joint_idx]
                p_acc_arr = plan["planned_acc"][:, self.joint_idx]
                self.all_vel_vals.extend(p_vel_arr)
                self.all_acc_vals.extend(p_acc_arr)
                local_max = p_t_arr.max()
                self.max_time_val = max(self.max_time_val, local_max)

        # Check overall max_time
        self.max_t_tick = self.tick_times.max() if self.tick_times.size > 0 else 0.0
        self.max_time = max(self.max_t_tick, self.max_time_val)

        if not self.all_pos_vals:
            self.all_pos_vals = [0.0]
        if not self.all_vel_vals:
            self.all_vel_vals = [0.0]
        if not self.all_acc_vals:
            self.all_acc_vals = [0.0]

        self.pos_min, self.pos_max = self._expand_range(
            min(self.all_pos_vals), max(self.all_pos_vals))
        self.vel_min, self.vel_max = self._expand_range(
            min(self.all_vel_vals), max(self.all_vel_vals))
        self.acc_min, self.acc_max = self._expand_range(
            min(self.all_acc_vals), max(self.all_acc_vals))

    def _expand_range(self, vmin, vmax, ratio=0.05):
        """Expand the given range by a certain ratio to give some margin."""
        if vmin == vmax:
            return (vmin - 1, vmax + 1)
        span = vmax - vmin
        return (vmin - ratio * span, vmax + ratio * span)

    def _init_figure(self):
        """
        Create the matplotlib figure, subplots, and line objects for
        position, velocity, and acceleration of the specified joint.
        """
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.axes[0].set_ylabel(f"Position (J{self.joint_idx+1})")
        self.axes[1].set_ylabel(f"Velocity (J{self.joint_idx+1})")
        self.axes[2].set_ylabel(f"Acceleration (J{self.joint_idx+1})")
        self.axes[2].set_xlabel("Time (s)")

        # 1) Reference
        (self.line_pos_ref,) = self.axes[0].plot([], [], 'k.', label='Ref Pos')
        (self.line_vel_ref,) = self.axes[1].plot([], [], 'k.', label='Ref Vel')
        (self.line_acc_ref,) = self.axes[2].plot([], [], 'k.', label='Ref Acc')

        # 2) Waypoints (blue)
        (self.line_pos_wp,)  = self.axes[0].plot([], [], 'bo', label='Waypoints', zorder=10)

        # 3) Planned Traj (green)
        (self.line_pos_plan,) = self.axes[0].plot([], [], 'g.', label='Planned Pos (Future)')
        (self.line_vel_plan,) = self.axes[1].plot([], [], 'g.', label='Planned Vel (Future)')
        (self.line_acc_plan,) = self.axes[2].plot([], [], 'g.', label='Planned Acc (Future)')

        # 4) Current state (red triangle)
        (self.line_pos_cur,) = self.axes[0].plot([], [], 'r^', label='Current Pos',
                                                 markersize=10, zorder=11)
        (self.line_vel_cur,) = self.axes[1].plot([], [], 'r^', label='Current Vel',
                                                 markersize=10)
        (self.line_acc_cur,) = self.axes[2].plot([], [], 'r^', label='Current Acc',
                                                 markersize=10)

        # Set axis ranges
        self.axes[0].set_xlim(0, self.max_time + 0.1)
        self.axes[0].set_ylim(self.pos_min, self.pos_max)
        self.axes[1].set_ylim(self.vel_min, self.vel_max)
        self.axes[2].set_ylim(self.acc_min, self.acc_max)

        for ax in self.axes:
            ax.grid(True)

        self.axes[0].legend(loc='upper right', fontsize='small')
        self.axes[1].legend(loc='upper right', fontsize='small')
        self.axes[2].legend(loc='upper right', fontsize='small')

    def _prepare_animation_frames(self):
        """
        Prepare the time array used for the animation frames.
        Larger num_frames => smoother animation => longer total time.
        """
        self.num_frames = 300
        self.frames = np.linspace(0, self.max_time, self.num_frames)

        # default interval in ms is 50, we multiply by slower_factor
        self.animation_interval = int(50 * self.slower_factor)

    def init_func(self):
        """
        Initialization function for FuncAnimation. 
        It returns the line objects that will be updated.
        """
        return (
            self.line_pos_ref, self.line_vel_ref, self.line_acc_ref,
            self.line_pos_cur, self.line_vel_cur, self.line_acc_cur,
            self.line_pos_wp,
            self.line_pos_plan, self.line_vel_plan, self.line_acc_plan
        )

    def update_plot(self, frame_t):
        """
        The update callback for FuncAnimation. Given a time 'frame_t',
        update all line objects accordingly.
        """
        # 1) Plot reference states (where time <= frame_t)
        if self.tick_times.size > 0:
            mask_tick = (self.tick_times <= frame_t)
            ref_t = self.tick_times[mask_tick]
            ref_pos = self.tick_joint_pos_ref[mask_tick, self.joint_idx]
            ref_vel = self.tick_joint_vel_ref[mask_tick, self.joint_idx]
            ref_acc = self.tick_joint_acc_ref[mask_tick, self.joint_idx]
            self.line_pos_ref.set_data(ref_t, ref_pos)
            self.line_vel_ref.set_data(ref_t, ref_vel)
            self.line_acc_ref.set_data(ref_t, ref_acc)
        else:
            self.line_pos_ref.set_data([], [])
            self.line_vel_ref.set_data([], [])
            self.line_acc_ref.set_data([], [])

        # 2) Current state (closest tick)
        if self.tick_times.size > 0:
            idx_closest = np.argmin(np.abs(self.tick_times - frame_t))
            cur_pos_val = self.tick_joint_pos_ref[idx_closest, self.joint_idx]
            cur_vel_val = self.tick_joint_vel_ref[idx_closest, self.joint_idx]
            cur_acc_val = self.tick_joint_acc_ref[idx_closest, self.joint_idx]
        else:
            cur_pos_val, cur_vel_val, cur_acc_val = 0, 0, 0

        self.line_pos_cur.set_data([frame_t], [cur_pos_val])
        self.line_vel_cur.set_data([frame_t], [cur_vel_val])
        self.line_acc_cur.set_data([frame_t], [cur_acc_val])

        # 3) Find the current active plan (start_time <= frame_t)
        active_plan = None
        for p in self.plans:
            if p["start_time"] <= frame_t:
                active_plan = p
            else:
                break

        # If no active plan, clear blue/green points
        if active_plan is None:
            self.line_pos_wp.set_data([], [])
            self.line_pos_plan.set_data([], [])
            self.line_vel_plan.set_data([], [])
            self.line_acc_plan.set_data([], [])
            self._update_xlim(frame_t)
            return (
                self.line_pos_ref, self.line_vel_ref, self.line_acc_ref,
                self.line_pos_cur, self.line_vel_cur, self.line_acc_cur,
                self.line_pos_wp,
                self.line_pos_plan, self.line_vel_plan, self.line_acc_plan
            )

        # 3a) Waypoints: show all for now (or only <= frame_t if needed)
        wp_t_arr = active_plan["waypoints_times"]
        wp_pos_arr = active_plan["waypoints_positions"]
        if wp_t_arr.size > 0:
            self.line_pos_wp.set_data(wp_t_arr, wp_pos_arr[:, self.joint_idx])
        else:
            self.line_pos_wp.set_data([], [])

        # 3b) Planned trajectory: only show future (time >= frame_t)
        plan_t_arr = active_plan["planned_times"]
        plan_p_arr = active_plan["planned_pos"]
        plan_v_arr = active_plan["planned_vel"]
        plan_a_arr = active_plan["planned_acc"]

        if plan_t_arr.size > 0:
            mask_future = (plan_t_arr >= frame_t)
            t_display = plan_t_arr[mask_future]
            p_display = plan_p_arr[mask_future, self.joint_idx]
            v_display = plan_v_arr[mask_future, self.joint_idx]
            a_display = plan_a_arr[mask_future, self.joint_idx]

            self.line_pos_plan.set_data(t_display, p_display)
            self.line_vel_plan.set_data(t_display, v_display)
            self.line_acc_plan.set_data(t_display, a_display)
        else:
            self.line_pos_plan.set_data([], [])
            self.line_vel_plan.set_data([], [])
            self.line_acc_plan.set_data([], [])

        # 4) Update x-axis range in a rolling window
        self._update_xlim(frame_t)

        return (
            self.line_pos_ref, self.line_vel_ref, self.line_acc_ref,
            self.line_pos_cur, self.line_vel_cur, self.line_acc_cur,
            self.line_pos_wp,
            self.line_pos_plan, self.line_vel_plan, self.line_acc_plan
        )

    def _update_xlim(self, frame_t):
        """
        Rolling time window, [frame_t - half_window, frame_t + half_window].
        This ensures that the red triangle (current time) is in the center,
        unless it hits 0 or max_time boundaries.
        """
        left = frame_t - self.half_window
        right = frame_t + self.half_window

        if left < 0:
            left = 0
            right = 2 * self.half_window
        if right > self.max_time:
            right = self.max_time
            left = self.max_time - 2 * self.half_window
            if left < 0:
                left = 0
                right = self.max_time

        for ax in self.axes:
            ax.set_xlim(left, right)

    def run(self):
        """
        Creates the FuncAnimation and starts the interactive animation.
        """
        self.ani = FuncAnimation(
            self.fig, 
            self.update_plot,
            frames=self.frames,
            init_func=self.init_func,
            blit=False,
            interval=self.animation_interval,
            repeat=False
        )
        plt.tight_layout()
        plt.show()


# ------------------- Example usage script (if run standalone) -------------------
if __name__ == "__main__":
    # Instantiate and run
    animator = JointTrajectoryAnimator(
        data_path="/home/yue/Workspace/wos/tmp/logdata.json",
        joint_idx=1,         # Visualize joint 
        half_window=1.0,     # half window
        slower_factor=1.0    # normal speed
    )
    animator.run()
