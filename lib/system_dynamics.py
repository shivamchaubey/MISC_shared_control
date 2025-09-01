import numpy as np

class SystemDynamics:
    def __init__(self, dt = 0.1, damping_factor = 0.1):
        self.dt = dt
        self.damping_factor = damping_factor
        self.system_dynamics()
        self.curr_state = []
        self.curr_input = []
        self.next_state = []

    def system_dynamics(self):
        # Define the system dynamics
        dt = self.dt
        # self.A = np.array([[1, 0, dt, 0],
        #             [0, 1, 0, dt],
        #             [0, 0, 1, 0],
        #             [0, 0, 0, 1]])
        # self.B = np.array([[0.5 * dt**2, 0],
        #             [0, 0.5 * dt**2],
        #             [dt, 0],
        #             [0, dt]])
        damping_fun = 1 - self.damping_factor * dt
        # damping_fun = (1 - self.damping_factor) * dt
        # damping_fun = np.exp(-self.damping_factor * dt)

        # State transition matrix with damping
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, damping_fun, 0],
                           [0, 0, 0, damping_fun]])
        
        # Control matrix
        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])
            
    def next(self, x, u):
        self.next_state = self.A @ x + self.B @ u
        self.curr_state = x
        self.curr_input = u
        return self.next_state