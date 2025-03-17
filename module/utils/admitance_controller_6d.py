class AdmittanceController6D:
    def __init__(self, M, B, K):
        assert len(M) == 6 and len(B) == 6 and len(K) == 6
        self.M = M[:]
        self.B = B[:]
        self.K = K[:]

        self.velocity = [0.0] * 6 

    def update(self, force_6d, dt):

        delta_x = [0.0] * 6
        new_velocity = [0.0] * 6

        for i in range(6):
            if abs(self.M[i]) < 1e-9:
                if abs(self.B[i]) < 1e-9:
                    new_velocity[i] = 0.0
                    delta_x[i] = 0.0
                else:
                    x_dot_i = (force_6d[i] - self.K[i] * 0) / self.B[i]
                    delta_x[i] = x_dot_i * dt
                    new_velocity[i] = x_dot_i
            else:
                x_ddot = (force_6d[i] - self.B[i] * self.velocity[i] - self.K[i] * 0) / self.M[i]
                x_dot_i = self.velocity[i] + x_ddot * dt
                delta_x[i] = x_dot_i * dt
                new_velocity[i] = x_dot_i

        self.velocity = new_velocity 
        return delta_x 

    def reset(self):
        self.position = [0.0] * 6
        self.velocity = [0.0] * 6
