"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
from scipy import stats
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors

class Observer:
    def __init__(self, ts_control, initial_measurements = MsgSensors()):
        # initialized estimated state message
        self.estimated_state = MsgState()
        # use alpha filters to low pass filter gyros and accels
        # alpha = Ts/(Ts + tau) where tau is the LPF time constant

        self.lpf_gyro_x = AlphaFilter(alpha=0.5, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.5, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.9, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=0, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0, y0=initial_measurements.accel_z)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.9, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.9, y0=initial_measurements.diff_pressure)
        # ekf for phi and theta
        self.attitude_ekf = EkfAttitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition()

    def update(self, measurement):
        ##### TODO #####
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x) - self.estimated_state.bx
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y) - self.estimated_state.by
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z) - self.estimated_state.bz

        # invert sensor model to get altitude and airspeed
        self.estimated_state.altitude = self.lpf_abs.update(measurement.abs_pressure) / (CTRL.rho * CTRL.gravity)
        self.estimated_state.Va = np.sqrt(2 * self.lpf_diff.update(measurement.diff_pressure) / CTRL.rho)

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(measurement, self.estimated_state)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(measurement, self.estimated_state)

        # not estimating these
        self.estimated_state.alpha = 0.0
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state


class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha * self.y + (1 - self.alpha) * u
        return self.y


class EkfAttitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        ##### TODO #####
        self.Q = np.diag([0.1, 0.2])
        self.Q_gyro = np.diag([SENSOR.gyro_sigma**2, SENSOR.gyro_sigma**2, SENSOR.gyro_sigma**2])
        self.R_accel = np.diag([SENSOR.accel_sigma**2, SENSOR.accel_sigma**2, SENSOR.accel_sigma**2])
        self.N = 2  # number of prediction step per sample
        self.xhat = np.array([[0.0], [0.0]]) # initial state: phi, theta
        self.P = np.diag([0, 0])
        self.Ts = SIM.ts_control/self.N
        self.gate_threshold = stats.chi2.isf(q=0.01, df=3)
        self.gate_threshold = 100000

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        phi = x.item(0)
        theta = x.item(1)
        # p = measurement.gyro_x - state.bx
        # q = measurement.gyro_y - state.by
        # r = measurement.gyro_z - state.bz
        # phi = state.phi
        # theta = state.theta
        p = state.p
        q = state.q
        r = state.r
        G = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0.0, np.cos(phi), -np.sin(phi)]])
        f_ = G @ np.array([[p], [q], [r]])
        return f_

    def h(self, x, measurement, state):
        # measurement model y
        phi = x.item(0)
        theta = x.item(1)
        # p = measurement.gyro_x - state.bx
        # q = measurement.gyro_y - state.by
        # r = measurement.gyro_z - state.bz
        # # Va = state.Va
        # Va = np.sqrt(2 * measurement.diff_pressure / CTRL.rho)
        # phi = state.phi
        # theta = state.theta
        p = state.p
        q = state.q
        r = state.r
        Va = state.Va
        G = np.array([[0.0, Va * np.sin(theta), 0.0, np.sin(theta)],
                      [-Va * np.sin(theta), 0.0, Va * np.cos(theta), -np.cos(theta) * np.sin(phi)],
                      [0.0, -Va * np.cos(theta), 0.0, -np.cos(theta) * np.cos(phi)]])
        h_ = G @ np.array([[p], [q], [r], [CTRL.gravity]])
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        Tp = self.Ts
        for i in range(0, self.N):
            self.xhat = self.xhat + Tp * self.f(self.xhat, measurement, state)
            A = jacobian(self.f, self.xhat, measurement, state)
            A_d = np.eye(2) + Tp * A + Tp**2 * A @ A
            phi = self.xhat.item(0)
            theta = self.xhat.item(1)
            G = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0.0, np.cos(phi), -np.sin(phi)]])
            self.P = A_d @ self.P @ A_d.T + Tp**2 * (G @ self.Q_gyro @ G.T + self.Q)
            # self.P = self.P + Tp / self.N * (A @ self.P + self.P @ A.T + self.Q)

    def measurement_update(self, measurement, state):
        # measurement updates
        h = self.h(self.xhat, measurement, state)
        C = jacobian(self.h, self.xhat, measurement, state)
        y = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T

        S_inv = np.linalg.inv(self.R_accel + C @ self.P @ C.T)
        if (y-h).T @ S_inv @ (y-h) < self.gate_threshold:
            L = self.P @ C.T @ S_inv
            tmp = np.eye(2) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R_accel @ L.T
            self.xhat = self.xhat + L @ (y - h)


class EkfPosition:
    # implement continous-discrete EKF to estimate pn, pe, Vg, chi, wn, we, psi
    def __init__(self):
        self.Q = np.diag([
                    0.001,  # pn
                    0.001,  # pe
                    0.00001,  # Vg
                    0.000559269, # chi
                    0.01, # wn
                    0.01, # we
                    0.000559269, #0.0001, # psi
                    ])
        self.R_gps = np.diag([
                    SENSOR.gps_n_sigma**2,  # y_gps_n
                    SENSOR.gps_e_sigma**2,  # y_gps_e
                    SENSOR.gps_Vg_sigma**2,  # y_gps_Vg
                    SENSOR.gps_course_sigma**2,  # y_gps_course
                    ])
        self.R_pseudo = np.diag([
                    0.01,  # pseudo measurement #1
                    0.01,  # pseudo measurement #2
                    ])
        self.N = 2  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        self.P = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.gps_n_old = 0
        self.gps_e_old = 0
        self.gps_Vg_old = 0
        self.gps_course_old = 0
        self.pseudo_threshold = stats.chi2.isf(q=0.01, df=4)
        # self.pseudo_threshold = 100000
        self.gps_threshold = 100000 # don't gate GPS

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.north = self.xhat.item(0)
        state.east = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        q = measurement.gyro_y - state.by
        r = measurement.gyro_z - state.bz
        # Va = np.sqrt(2 * measurement.diff_pressure / CTRL.rho)
        # q = state.q
        # r = state.r
        Va = state.Va
        if x.item(2) == 0:
            x[2, 0] = state.Vg
        psidot = q * np.sin(state.phi) / np.cos(state.theta) + r * np.cos(state.phi) / np.cos(state.theta)
        f_ = np.array([[x.item(2) * np.cos(x.item(3))],
                       [x.item(2) * np.sin(x.item(3))],
                       [Va / x.item(2) * psidot * (x.item(5) * np.cos(x.item(6) - x.item(4) * np.sin(x.item(6))))], 
                       [CTRL.gravity / x.item(2) * np.tan(state.phi)],
                       [0.0],
                       [0.0],
                       [q * np.sin(state.phi) / np.cos(state.theta) + r * np.cos(state.phi) / np.cos(state.theta)],
                       ])
        return f_

    def h_gps(self, x, measurement, state):
        # measurement model for gps measurements
        h_ = np.array([
            [measurement.gps_n], #pn
            [measurement.gps_e], #pe
            [measurement.gps_Vg], #Vg
            [measurement.gps_course], #chi
            # [state.north],
            # [state.east],
            # [state.Vg],
            # [state.chi]
        ])
        return h_

    def h_pseudo(self, x, measurement, state):
        # measurement model for wind triangle pseudo measurement
        # Va = np.sqrt(2 * measurement.diff_pressure / CTRL.rho)
        Va = state.Va
        h_ = np.array([
            [Va * np.cos(x.item(6) + x.item(4) - x.item(2) * np.cos(x.item(3)))],  # wind triangle x
            [Va * np.sin(x.item(6) + x.item(5) - x.item(2) * np.sin(x.item(3)))],  # wind triangle y
        ])
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat = self.xhat + self.Ts * self.f(self.xhat, measurement, state)

            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)
            
            # convert to discrete time models
            A_d = np.eye(7) + self.Ts * A + self.Ts**2 * A @ A
            
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2 * self.Q

    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, measurement, state)
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        y = np.array([[0, 0]]).T
        S_inv = np.linalg.inv(self.R_pseudo + C @ self.P @ C.T)
        if (y-h).T @ S_inv @ (y-h) < self.pseudo_threshold:
            L = self.P @ C.T @ S_inv
            tmp = np.eye(7) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R_pseudo @ L.T
            self.xhat = self.xhat + L @ (y - h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, measurement, state)
            C = jacobian(self.h_gps, self.xhat, measurement, state)
            y_chi = wrap(measurement.gps_course, h[3, 0])
            y = np.array([[measurement.gps_n,
                           measurement.gps_e,
                           measurement.gps_Vg,
                           y_chi]]).T
            S_inv = np.linalg.inv(self.R_gps + C @ self.P @ C.T)
            if (y-h).T @ S_inv @ (y-h) < self.gps_threshold:
                L = self.P @ C.T @ S_inv
                tmp = np.eye(7) - L @ C
                self.P = tmp @ self.P @ tmp.T + L @ self.R_gps @ L.T
                self.xhat = self.xhat + L @ (y - h)

            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course


def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, measurement, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J