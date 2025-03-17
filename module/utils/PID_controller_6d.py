class PIDController6D:
    def __init__(self, kp, ki, kd, output_limits=None, setpoint=[0.0,0.0,0.0,0.0,0.0,0.0]):
        """
        Initialize a 6D PID controller.

        :param kp: List of proportional gains (length 6)
        :param ki: List of integral gains (length 6)
        :param kd: List of derivative gains (length 6)
        :param setpoint: Target values for each dimension (default: [0, 0, 0, 0, 0, 0])
        :param output_limits: List of (min, max) tuples for each dimension (default: no limits)
        """
        self.kp = kp[:]  # Copy list to avoid modifying original
        self.ki = ki[:]
        self.kd = kd[:]

        self.setpoint = setpoint
        self.output_limits = output_limits if output_limits else [(None, None)] * 6

        self.prev_error = [0.0] * 6  # Store previous error
        self.integral = [0.0] * 6    # Integral term

    def update(self, current_value, dt):
        """
        Compute the new 6D control output.

        :param current_value: List of current measured values (length 6)
        :param dt: Time interval (seconds)
        :return: List of PID control outputs (length 6)
        """
        error = [self.setpoint[i] - current_value[i] for i in range(6)]

        # Compute P, I, and D terms
        P = [self.kp[i] * error[i] for i in range(6)]
        self.integral = [self.integral[i] + error[i] * dt for i in range(6)]
        I = [self.ki[i] * self.integral[i] for i in range(6)]
        D = [
            self.kd[i] * ((error[i] - self.prev_error[i]) / dt) if dt > 0 else 0.0
            for i in range(6)
        ]

        # Calculate output
        output = [P[i] + I[i] + D[i] for i in range(6)]

        # Apply output limits for each dimension
        for i in range(6):
            min_out, max_out = self.output_limits[i]
            if min_out is not None:
                output[i] = max(min_out, output[i])
            if max_out is not None:
                output[i] = min(max_out, output[i])

        # Update stored values
        self.prev_error = error[:]

        return output  # Return as a standard Python list

    def reset(self):
        """ Reset the PID controller state for all dimensions. """
        self.prev_error = [0.0] * 6
        self.integral = [0.0] * 6
