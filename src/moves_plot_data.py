import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_log_data(filepath):
    """Load and parse logdata.json"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

class LogDataAnimator:
    def __init__(self, log_data, dim=0):
        """
        :param log_data: Parsed data from logdata.json
        :param dim:      The specific dimension (joint/coordinate index) to visualize, default is 0
        """
        self.log_data = log_data
        self.data_points = log_data["data_points"]
        self.dim = dim  # Only plot this dimension

        # Create a figure with 3 subplots: position, velocity, acceleration
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        self.fig.suptitle(f"Arm Realtime Move Log (Dimension={dim})")

        self.axs[0].set_ylabel("Position")
        self.axs[1].set_ylabel("Velocity")
        self.axs[2].set_ylabel("Acceleration")
        self.axs[2].set_xlabel("Time (s)")

        # -- Line for past trajectory --
        (self.pos_line_past,) = self.axs[0].plot([], [], color='black', lw=1, label='past pos')
        (self.vel_line_past,) = self.axs[1].plot([], [], color='black', lw=1, label='past vel')
        (self.acc_line_past,) = self.axs[2].plot([], [], color='black', lw=1, label='past acc')

        # -- Future trajectory as small black dots --
        self.pos_future_scatter = self.axs[0].scatter([], [], marker='.', color='black', s=20)
        self.vel_future_scatter = self.axs[1].scatter([], [], marker='.', color='black', s=20)
        self.acc_future_scatter = self.axs[2].scatter([], [], marker='.', color='black', s=20)

        # -- Current reference (red triangle) & buffer targets (blue circles) --
        self.current_ref_scatter = self.axs[0].scatter([], [], marker='^', color='red', s=60)
        self.buffer_targets_scatter = self.axs[0].scatter([], [], marker='o', color='blue', s=40)

        # Pre-compute y-axis limits by scanning all data
        (self.pos_min, self.pos_max,
         self.vel_min, self.vel_max,
         self.acc_min, self.acc_max) = self._compute_data_range()

        # Add margins to avoid flat lines
        def add_margin(vmin, vmax, margin_ratio=0.1):
            if vmin == float('inf') or vmax == float('-inf'):
                # If data is empty, set a default range
                return -1, 1
            if np.isclose(vmin, vmax):
                # Avoid too narrow range
                delta = abs(vmax) * margin_ratio + 1e-3
                return vmin - delta, vmax + delta
            delta = (vmax - vmin) * margin_ratio
            return vmin - delta, vmax + delta

        # Compute y-axis range for each subplot
        pmin, pmax = add_margin(self.pos_min, self.pos_max)
        vmin, vmax = add_margin(self.vel_min, self.vel_max)
        amin, amax = add_margin(self.acc_min, self.acc_max)

        self.axs[0].set_ylim(pmin, pmax)
        self.axs[1].set_ylim(vmin, vmax)
        self.axs[2].set_ylim(amin, amax)

        self.frame_idx = 0
        self.max_frames = len(self.data_points)

        # View window size in seconds, centered around current time
        self.window_size = 2.0
        self.half_window = self.window_size / 2

    def _compute_data_range(self):
        """
        Iterate through all data_points to extract global min/max
        for position, velocity, and acceleration across:
        - past_trajectory
        - buffer_trajectory
        - current_ref
        - buffer_targets
        """
        pos_min, pos_max = float('inf'), float('-inf')
        vel_min, vel_max = float('inf'), float('-inf')
        acc_min, acc_max = float('inf'), float('-inf')

        for dp in self.data_points:
            # past_trajectory
            for pt in dp["past_trajectory"]:
                st = pt["state"]
                p = st["position"][self.dim]
                v = st["velocity"][self.dim]
                a = st["acceleration"][self.dim]
                pos_min = min(pos_min, p)
                pos_max = max(pos_max, p)
                vel_min = min(vel_min, v)
                vel_max = max(vel_max, v)
                acc_min = min(acc_min, a)
                acc_max = max(acc_max, a)

            # buffer_trajectory
            for bt in dp["buffer_trajectory"]:
                st = bt["state"]
                p = st["position"][self.dim]
                v = st["velocity"][self.dim]
                a = st["acceleration"][self.dim]
                pos_min = min(pos_min, p)
                pos_max = max(pos_max, p)
                vel_min = min(vel_min, v)
                vel_max = max(vel_max, v)
                acc_min = min(acc_min, a)
                acc_max = max(acc_max, a)

            # current_ref
            cref = dp["current_ref"]
            if "position" in cref and len(cref["position"]) > self.dim:
                p = cref["position"][self.dim]
                v = cref["velocity"][self.dim]
                a = cref["acceleration"][self.dim]
                pos_min = min(pos_min, p)
                pos_max = max(pos_max, p)
                vel_min = min(vel_min, v)
                vel_max = max(vel_max, v)
                acc_min = min(acc_min, a)
                acc_max = max(acc_max, a)

            # buffer_targets (only position is used)
            if "buffer_targets" in dp:
                for bpos in dp["buffer_targets"]:
                    if len(bpos) > self.dim:
                        p = bpos[self.dim]
                        pos_min = min(pos_min, p)
                        pos_max = max(pos_max, p)

        return (pos_min, pos_max, vel_min, vel_max, acc_min, acc_max)

    def init_animation(self):
        # Clear past trajectory (lines)
        self.pos_line_past.set_data([], [])
        self.vel_line_past.set_data([], [])
        self.acc_line_past.set_data([], [])

        # Clear future trajectory (scatters) with empty Nx2 arrays
        self.pos_future_scatter.set_offsets(np.empty((0, 2)))
        self.vel_future_scatter.set_offsets(np.empty((0, 2)))
        self.acc_future_scatter.set_offsets(np.empty((0, 2)))

        # Clear current_ref and buffer_targets
        self.current_ref_scatter.set_offsets(np.empty((0, 2)))
        self.buffer_targets_scatter.set_offsets(np.empty((0, 2)))

        return []

    def update_animation(self, frame):
        if frame >= self.max_frames:
            return []

        dp = self.data_points[frame]
        t_now = float(dp["time_from_start"])

        # === Past Trajectory (black line) ===
        past_time, past_pos, past_vel, past_acc = [], [], [], []
        for pt in dp["past_trajectory"]:
            t_pt = float(pt["time_from_start"]) * 1e-9
            if (t_pt < t_now - self.half_window) or (t_pt > t_now + self.half_window):
                continue
            st = pt["state"]
            past_time.append(t_pt)
            past_pos.append(st["position"][self.dim])
            past_vel.append(st["velocity"][self.dim])
            past_acc.append(st["acceleration"][self.dim])

        self.pos_line_past.set_data(past_time, past_pos)
        self.vel_line_past.set_data(past_time, past_vel)
        self.acc_line_past.set_data(past_time, past_acc)

        # === Future Trajectory (black scatter) ===
        ftime, fpos, fvel, facc = [], [], [], []
        for bt in dp["buffer_trajectory"]:
            t_bt = float(bt["time_from_start"]) * 1e-9
            actual_time = t_now + t_bt
            if (actual_time < t_now - self.half_window) or (actual_time > t_now + self.half_window):
                continue
            st = bt["state"]
            ftime.append(actual_time)
            fpos.append(st["position"][self.dim])
            fvel.append(st["velocity"][self.dim])
            facc.append(st["acceleration"][self.dim])

        # Convert to Nx2 arrays
        if len(ftime) == 0:
            pos_future_data = np.empty((0, 2))
            vel_future_data = np.empty((0, 2))
            acc_future_data = np.empty((0, 2))
        else:
            pos_future_data = np.column_stack((ftime, fpos))
            vel_future_data = np.column_stack((ftime, fvel))
            acc_future_data = np.column_stack((ftime, facc))

        self.pos_future_scatter.set_offsets(pos_future_data)
        self.vel_future_scatter.set_offsets(vel_future_data)
        self.acc_future_scatter.set_offsets(acc_future_data)

        # === Current Reference (red triangle) ===
        current_ref = dp["current_ref"]
        if ("position" in current_ref) and len(current_ref["position"]) > self.dim:
            curr_p = current_ref["position"][self.dim]
            current_offsets = np.array([[t_now, curr_p]])
        else:
            current_offsets = np.empty((0, 2))
        self.current_ref_scatter.set_offsets(current_offsets)

        # === Buffer Targets (blue circles) ===
        buf_data = []
        if "buffer_targets" in dp:
            for bpos in dp["buffer_targets"]:
                if len(bpos) > self.dim:
                    buf_data.append([t_now, bpos[self.dim]])
        if len(buf_data) == 0:
            buffer_offsets = np.empty((0, 2))
        else:
            buffer_offsets = np.array(buf_data)

        self.buffer_targets_scatter.set_offsets(buffer_offsets)

        # Keep current time centered in the view window
        for ax in self.axs:
            ax.set_xlim(t_now - self.half_window, t_now + self.half_window)

        return []

    def animate(self):
        ani = FuncAnimation(
            self.fig,
            self.update_animation,
            frames=range(self.max_frames),
            init_func=self.init_animation,
            interval=100,
            blit=False
        )
        plt.show()

def main():
    # Default log file path
    filepath = "/home/yue/Workspace/wos/tmp/logdata.json"

    # Optional command line argument: dimension index
    # Example: python plot_data.py 2 => dim=2
    dim = 0
    if len(sys.argv) > 1:
        try:
            dim = int(sys.argv[1])
        except ValueError:
            print("Usage: python plot_data.py [dim]")
            print("dim must be an integer. Default is 0.")
            sys.exit(1)

    log_data = load_log_data(filepath)
    animator = LogDataAnimator(log_data, dim=dim)
    animator.animate()

if __name__ == "__main__":
    main()
