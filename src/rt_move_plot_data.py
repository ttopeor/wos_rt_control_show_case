#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DimensionTrajectoryAnimator:
    """
    A class to animate either a specific joint dimension or a specific cartesian
    dimension (x,y,z,roll,pitch,yaw) from a given JSON log file containing
    'ticks' and 'set_target' data.

    The animation will display (for one dimension only):
      - Position (top subplot)
      - Velocity (middle subplot)
      - Acceleration (bottom subplot)

    Depending on space_mode:
      * 'joint':
          dimension_show = 0..6 (or however many joints you have)
          => read from 'current_joint_state_ref/current_joint_state'
             'joints_waypoints', 'planned_joints_trajectory'
      * 'cartesian':
          dimension_show = 0..5 => x=0,y=1,z=2,roll=3,pitch=4,yaw=5
          => read from 'current_cartesian_state_ref/current_cartesian_state'
             'request_waypoints', 'planned_cartesian_trajectory'
    """

    def __init__(self,
                 data_path: str,
                 space_mode: str = 'joint',
                 dimension_show: int = 0,
                 half_window: float = 0.4,
                 slower_factor: float = 1.0):
        """
        Parameters
        ----------
        data_path : str
            Path to the JSON log file.
        space_mode : str
            'joint' or 'cartesian'
        dimension_show : int
            Which dimension to display.
            If 'joint' => dimension_show is the joint index (0..6 or so).
            If 'cartesian' => 0..5 => x=0,y=1,z=2,roll=3,pitch=4,yaw=5
        half_window : float
            Rolling time window half-size (in seconds)
        slower_factor : float
            Speed factor for animation interval
        """
        self.data_path = data_path
        self.space_mode = space_mode.lower()
        self.dimension_show = dimension_show
        self.half_window = half_window
        self.slower_factor = slower_factor

        # Load JSON
        with open(self.data_path, 'r') as f:
            log_data = json.load(f)

        # Retrieve tick logs
        self.ticks = log_data.get("ticks", [])
        # Retrieve set_target logs
        self.set_targets = log_data.get("set_target", [])

        # Parse ticks & set_target
        self._parse_tick_data()
        self._parse_set_target_data()

        # Compute overall ranges
        self._compute_plot_ranges()

        # Prepare figure & lines
        self._init_figure()

        # Prepare frames (times)
        self._prepare_animation_frames()

        # 是否正在播放动画
        self.anim_running = True
        # 当前帧索引（给手动前/后翻页用）
        self.current_frame_idx = 0

        self.ani = None

    def _parse_tick_data(self):
        """Parse 'ticks' logs, picking the correct fields depending on space_mode."""
        self.tick_times = []
        self.tick_pos_ref = []
        self.tick_vel_ref = []
        self.tick_acc_ref = []

        self.tick_pos_cur = []
        self.tick_vel_cur = []
        self.tick_acc_cur = []

        # Depending on space_mode, we pick different JSON keys
        pos_ref_key = "current_joint_state_ref"
        pos_cur_key = "current_joint_state"

        if self.space_mode == 'cartesian':
            pos_ref_key = "current_cartesian_state_ref"
            pos_cur_key = "current_cartesian_state"

        for tk in self.ticks:
            t = tk["time_from_start"]
            self.tick_times.append(t)

            ref_state = tk[pos_ref_key]
            cur_state = tk[pos_cur_key]

            self.tick_pos_ref.append(ref_state["position"])
            self.tick_vel_ref.append(ref_state["velocity"])
            self.tick_acc_ref.append(ref_state["acceleration"])

            self.tick_pos_cur.append(cur_state["position"])
            self.tick_vel_cur.append(cur_state["velocity"])
            self.tick_acc_cur.append(cur_state["acceleration"])

        self.tick_times = np.array(self.tick_times)
        self.tick_pos_ref = np.array(self.tick_pos_ref)
        self.tick_vel_ref = np.array(self.tick_vel_ref)
        self.tick_acc_ref = np.array(self.tick_acc_ref)

        self.tick_pos_cur = np.array(self.tick_pos_cur)
        self.tick_vel_cur = np.array(self.tick_vel_cur)
        self.tick_acc_cur = np.array(self.tick_acc_cur)

    def _parse_set_target_data(self):
        """
        Parse the 'set_target' logs. If space_mode='joint', read 'joints_waypoints'
        & 'planned_joints_trajectory'. If 'cartesian', read 'request_waypoints'
        & 'planned_cartesian_trajectory'.
        """
        self.plans = []
        if self.space_mode == 'cartesian':
            wp_key = "request_waypoints"
            plan_key = "planned_cartesian_trajectory"
        else:
            wp_key = "joints_waypoints"
            plan_key = "planned_joints_trajectory"

        for st in self.set_targets:
            st_time = st["time_from_start"]

            waypoints = st.get(wp_key, [])
            wp_times_local = []
            wp_pos_local = []

            cum_time = st_time
            for wp in waypoints:
                cum_time += wp["duration"]
                wp_times_local.append(cum_time)
                wp_pos_local.append(wp["position"])

            wp_times_local = np.array(wp_times_local)
            wp_pos_local = np.array(wp_pos_local)

            planned_traj = st.get(plan_key, [])
            planned_times_local = []
            planned_pos_local = []
            planned_vel_local = []
            planned_acc_local = []

            for pt in planned_traj:
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
        Only storing min/max for the dimension_show index across position/velocity/acceleration.
        """
        self.all_pos_vals = []
        self.all_vel_vals = []
        self.all_acc_vals = []

        # Ticks
        if len(self.ticks) > 0:
            pos_ref_dim = self.tick_pos_ref[:, self.dimension_show]
            pos_cur_dim = self.tick_pos_cur[:, self.dimension_show]
            vel_ref_dim = self.tick_vel_ref[:, self.dimension_show]
            vel_cur_dim = self.tick_vel_cur[:, self.dimension_show]
            acc_ref_dim = self.tick_acc_ref[:, self.dimension_show]
            acc_cur_dim = self.tick_acc_cur[:, self.dimension_show]

            self.all_pos_vals.extend(pos_ref_dim)
            self.all_pos_vals.extend(pos_cur_dim)
            self.all_vel_vals.extend(vel_ref_dim)
            self.all_vel_vals.extend(vel_cur_dim)
            self.all_acc_vals.extend(acc_ref_dim)
            self.all_acc_vals.extend(acc_cur_dim)

            self.max_t_tick = self.tick_times.max()
        else:
            self.max_t_tick = 0

        self.max_time_val = 0
        # Plans
        for plan in self.plans:
            wp_t_arr = plan["waypoints_times"]
            wp_p_arr = plan["waypoints_positions"]
            if wp_t_arr.size > 0:
                self.max_time_val = max(self.max_time_val, wp_t_arr.max())
                if wp_p_arr.shape[1] > self.dimension_show:
                    self.all_pos_vals.extend(wp_p_arr[:, self.dimension_show])

            plan_t_arr = plan["planned_times"]
            plan_p_arr = plan["planned_pos"]
            plan_v_arr = plan["planned_vel"]
            plan_a_arr = plan["planned_acc"]
            if plan_t_arr.size > 0:
                self.max_time_val = max(self.max_time_val, plan_t_arr.max())
                if plan_p_arr.shape[1] > self.dimension_show:
                    self.all_pos_vals.extend(plan_p_arr[:, self.dimension_show])
                if plan_v_arr.shape[1] > self.dimension_show:
                    self.all_vel_vals.extend(plan_v_arr[:, self.dimension_show])
                if plan_a_arr.shape[1] > self.dimension_show:
                    self.all_acc_vals.extend(plan_a_arr[:, self.dimension_show])

        self.max_time = self.max_time_val + 0.1

        if not self.all_pos_vals:
            self.all_pos_vals = [0.]
        if not self.all_vel_vals:
            self.all_vel_vals = [0.]
        if not self.all_acc_vals:
            self.all_acc_vals = [0.]

        self.pos_min, self.pos_max = self._expand_range(min(self.all_pos_vals), max(self.all_pos_vals))
        self.vel_min, self.vel_max = self._expand_range(min(self.all_vel_vals), max(self.all_vel_vals))
        self.acc_min, self.acc_max = self._expand_range(min(self.all_acc_vals), max(self.all_acc_vals))

    def _expand_range(self, vmin, vmax, ratio=0.05):
        if vmin == vmax:
            return (vmin - 1, vmax + 1)
        span = vmax - vmin
        return (vmin - ratio*span, vmax + ratio*span)

    def _init_figure(self):
        """
        We'll just name the subplots generically: "Position (Dim=dimension_show)",
        "Velocity (Dim=dimension_show)", etc. 
        If in cartesian mode and dimension_show=0 => x dimension,
        1 => y dimension, etc.
        """
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Title or axis label
        if self.space_mode == 'cartesian':
            label_prefix = f"Cartesian Dim {self.dimension_show}"
        else:
            label_prefix = f"Joint {self.dimension_show}"

        self.axes[0].set_ylabel(f"{label_prefix} Pos")
        self.axes[1].set_ylabel(f"{label_prefix} Vel")
        self.axes[2].set_ylabel(f"{label_prefix} Acc")
        self.axes[2].set_xlabel("Time (s)")

        # Create line objects
        (self.line_pos_ref,) = self.axes[0].plot([], [], 'k.', label='Ref Pos')
        (self.line_vel_ref,) = self.axes[1].plot([], [], 'k.', label='Ref Vel')
        (self.line_acc_ref,) = self.axes[2].plot([], [], 'k.', label='Ref Acc')

        (self.line_pos_plan,) = self.axes[0].plot([], [], 'g.', label='Plan Pos')
        (self.line_vel_plan,) = self.axes[1].plot([], [], 'g.', label='Plan Vel')
        (self.line_acc_plan,) = self.axes[2].plot([], [], 'g.', label='Plan Acc')

        (self.line_pos_wp,)  = self.axes[0].plot([], [], 'bo', label='Waypoints')

        (self.line_pos_cur,) = self.axes[0].plot([], [], 'r^', label='Current Pos', markersize=10)
        (self.line_vel_cur,) = self.axes[1].plot([], [], 'r^', label='Current Vel', markersize=10)
        (self.line_acc_cur,) = self.axes[2].plot([], [], 'r^', label='Current Acc', markersize=10)

        (self.line_pos_cur_track,) = self.axes[0].plot([], [], '--r', label='Current Pos Track')

        # Set axis range
        self.axes[0].set_xlim(0, self.max_time + 0.1)
        self.axes[0].set_ylim(self.pos_min, self.pos_max)
        self.axes[1].set_ylim(self.vel_min, self.vel_max)
        self.axes[2].set_ylim(self.acc_min, self.acc_max)

        for ax in self.axes:
            ax.grid(True)

        # Legend
        leg0 = self.axes[0].legend(loc='upper right', fontsize='small')
        leg1 = self.axes[1].legend(loc='upper right', fontsize='small')
        leg2 = self.axes[2].legend(loc='upper right', fontsize='small')
        leg0.set_zorder(12)
        leg1.set_zorder(12)
        leg2.set_zorder(12)

    def _prepare_animation_frames(self):
        self.num_frames = 300
        self.frames = np.linspace(0, self.max_time, self.num_frames)
        self.animation_interval = int(50 * self.slower_factor)

    def init_func(self):
        return (
            self.line_pos_ref, self.line_vel_ref, self.line_acc_ref,
            self.line_pos_cur_track,
            self.line_pos_cur, self.line_vel_cur, self.line_acc_cur,
            self.line_pos_wp,
            self.line_pos_plan, self.line_vel_plan, self.line_acc_plan
        )

    def update_plot(self, frame_t):
        """
        Called by FuncAnimation or manually from _on_key_press to update the plot
        at a specific time frame_t.
        """
        # 1) Plot reference states up to frame_t
        if self.tick_times.size > 0:
            mask_tick = (self.tick_times <= frame_t)
            ref_t = self.tick_times[mask_tick]

            pos_ref_dim = self.tick_pos_ref[mask_tick, self.dimension_show]
            vel_ref_dim = self.tick_vel_ref[mask_tick, self.dimension_show]
            acc_ref_dim = self.tick_acc_ref[mask_tick, self.dimension_show]

            self.line_pos_ref.set_data(ref_t, pos_ref_dim)
            self.line_vel_ref.set_data(ref_t, vel_ref_dim)
            self.line_acc_ref.set_data(ref_t, acc_ref_dim)
        else:
            self.line_pos_ref.set_data([], [])
            self.line_vel_ref.set_data([], [])
            self.line_acc_ref.set_data([], [])

        # 2) Current states (the last available tick up to frame_t)
        if self.tick_times.size > 0:
            mask_tick = (self.tick_times <= frame_t)
            ref_t = self.tick_times[mask_tick]

            pos_cur_dim_all = self.tick_pos_cur[mask_tick, self.dimension_show]
            self.line_pos_cur_track.set_data(ref_t, pos_cur_dim_all)

            if np.any(mask_tick):
                idx_last = np.where(mask_tick)[0][-1]
                pos_last = self.tick_pos_cur[idx_last, self.dimension_show]
                vel_last = self.tick_vel_ref[idx_last, self.dimension_show]
                acc_last = self.tick_acc_ref[idx_last, self.dimension_show]

                t_last = self.tick_times[idx_last]
                self.line_pos_cur.set_data([t_last], [pos_last])
                self.line_vel_cur.set_data([t_last], [vel_last])
                self.line_acc_cur.set_data([t_last], [acc_last])
            else:
                self.line_pos_cur.set_data([], [])
                self.line_vel_cur.set_data([], [])
                self.line_acc_cur.set_data([], [])
        else:
            # 若没有ticks
            self.line_pos_cur_track.set_data([], [])
            self.line_pos_cur.set_data([], [])
            self.line_vel_cur.set_data([], [])
            self.line_acc_cur.set_data([], [])

        # 3) Active plan
        active_plan = None
        for p in self.plans:
            if p["start_time"] <= frame_t:
                active_plan = p
            else:
                break

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

        # Waypoints
        wp_t_arr = active_plan["waypoints_times"]
        wp_p_arr = active_plan["waypoints_positions"]
        if wp_t_arr.size > 0 and wp_p_arr.shape[1] > self.dimension_show:
            wp_dim_val = wp_p_arr[:, self.dimension_show]
            self.line_pos_wp.set_data(wp_t_arr, wp_dim_val)
        else:
            self.line_pos_wp.set_data([], [])

        # Planned future
        plan_t_arr = active_plan["planned_times"]
        plan_p_arr = active_plan["planned_pos"]
        plan_v_arr = active_plan["planned_vel"]
        plan_a_arr = active_plan["planned_acc"]

        if plan_t_arr.size > 0 and plan_p_arr.shape[1] > self.dimension_show:
            mask_future = (plan_t_arr >= frame_t)
            t_display = plan_t_arr[mask_future]

            p_val = plan_p_arr[mask_future, self.dimension_show]
            self.line_pos_plan.set_data(t_display, p_val)

            if plan_v_arr.shape[1] > self.dimension_show:
                v_val = plan_v_arr[mask_future, self.dimension_show]
                self.line_vel_plan.set_data(t_display, v_val)
            else:
                self.line_vel_plan.set_data([], [])

            if plan_a_arr.shape[1] > self.dimension_show:
                a_val = plan_a_arr[mask_future, self.dimension_show]
                self.line_acc_plan.set_data(t_display, a_val)
            else:
                self.line_acc_plan.set_data([], [])
        else:
            self.line_pos_plan.set_data([], [])
            self.line_vel_plan.set_data([], [])
            self.line_acc_plan.set_data([], [])

        # 4) rolling window
        self._update_xlim(frame_t)

        return (
            self.line_pos_ref, self.line_vel_ref, self.line_acc_ref,
            self.line_pos_cur, self.line_vel_cur, self.line_acc_cur,
            self.line_pos_wp,
            self.line_pos_plan, self.line_vel_plan, self.line_acc_plan
        )

    def _update_xlim(self, frame_t):
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

    def _on_key_press(self, event):
        # p => pause/play
        if event.key == 'p':
            if self.anim_running:
                self.ani.event_source.stop()
                self.anim_running = False
            else:
                self.ani.event_source.start()
                self.anim_running = True

        # r => restart from beginning
        elif event.key == 'r':
            self.ani.event_source.stop()
            self.anim_running = True
            self.current_frame_idx = 0
            # 重新设置动画序列
            self.ani.frame_seq = self.ani.new_frame_seq()
            self.ani.event_source.start()

        # left => go to previous frame manually
        elif event.key == 'left':
            # 如果在播放，先暂停
            if self.anim_running:
                self.ani.event_source.stop()
                self.anim_running = False

            self.current_frame_idx = max(0, self.current_frame_idx - 1)
            frame_t = self.frames[self.current_frame_idx]
            self.update_plot(frame_t)
            plt.draw()

        # right => go to next frame manually
        elif event.key == 'right':
            # 如果在播放，先暂停
            if self.anim_running:
                self.ani.event_source.stop()
                self.anim_running = False

            self.current_frame_idx = min(self.num_frames - 1, self.current_frame_idx + 1)
            frame_t = self.frames[self.current_frame_idx]
            self.update_plot(frame_t)
            plt.draw()

    def run(self):
        """Run the interactive animation via FuncAnimation."""
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=self.frames,
            init_func=self.init_func,
            blit=False,
            interval=self.animation_interval,
            repeat=False
        )
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    animator = DimensionTrajectoryAnimator(
        data_path="/home/yue/Workspace/wos/tmp/6b805274-97a9-425c-b103-2505f5490788_logdata.json",
        #data_path="/home/yue/Workspace/wos_rt_control_show_case/data/ybot.json",
        space_mode='joint',  # cartesian or joint
        dimension_show=1,    # e.g. joint index=3
        half_window=3.0,
        slower_factor=1.0
    )
    animator.run()
