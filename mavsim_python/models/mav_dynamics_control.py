"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta

import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               [0],   # (13)
                               [0],   # (14)
                               ])
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.true_state = MsgState()
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state[0:13], forces_moments)
        k2 = self._derivatives(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # Extract the States
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)

        # Extract Forces/Moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # Position Kinematics
        pos_dot = Quaternion2Rotation(state[6:10]) @ state[3:6]

        # Position Dynamics
        u_dot = np.array([[r * v - q * w, p * w - r * u, q * u - p * v]]).T + (1 / MAV.mass) * np.array([[fx, fy, fz]]).T

        # rotational kinematics
        e0_dot = 0.5 * np.array([[0, -p, -q, -r], [p, 0, r, -q], [q, -r, 0, p], [r, q, -p, 0]]) @ np.array([[e0, e1, e2, e3]]).T

        # rotatonal dynamics
        p_dot = np.array([[MAV.gamma1 * p * q - MAV.gamma2 * q * r, MAV.gamma5 * p * r - MAV.gamma6 * (p**2 - r**2), MAV.gamma7 * p * q - MAV.gamma1 * q * r]]).T + \
                np.array([[MAV.gamma3 * l + MAV.gamma4 * n, 1 / MAV.Jy * m, MAV.gamma4 * l + MAV.gamma8 * n]]).T

        # collect the derivative of the states
        x_dot = np.array([[pos_dot.item(0), pos_dot.item(1), pos_dot.item(2), u_dot.item(0), u_dot.item(1), u_dot.item(2), e0_dot.item(0), e0_dot.item(1), e0_dot.item(2), e0_dot.item(3), p_dot.item(0), p_dot.item(1), p_dot.item(2)]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        # convert wind vector from world to body frame (self._wind = ?)
        R = Quaternion2Rotation(self._state[6:10])
        self._wind = R @ steady_state + gust

        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        ur, vr, wr = (self._state[3:6] - self._wind)[:, 0]
        # compute airspeed (self._Va = ?)
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)
        # compute angle of attack (self._alpha = ?)
        self._alpha = np.arctan2(wr, ur)
        # compute sideslip angle (self._beta = ?)
        self._beta = np.arcsin(vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        # extract states (phi, theta, psi, p, q, r)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        fg = MAV.mass * MAV.gravity * np.array([[2 * (e1 * e3 - e2 * e0)], [2 * (e2 * e3 + e1 * e0)], [e3**2 + e0**2 - e1**2 - e2**2]])

        # compute Lift and Drag coefficients (CL, CD)
        eq4_10_p1 = np.exp(-MAV.M * (self._alpha - MAV.alpha0))
        eq4_10_p2 = np.exp(MAV.M * (self._alpha + MAV.alpha0))
        sigma = (1 + eq4_10_p1 + eq4_10_p2) / ((1 + eq4_10_p1) * (1 + eq4_10_p2))
        CL = (1 - sigma) * (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) + sigma * (2 * np.sign(self._alpha) * np.sin(self._alpha)**2 * np.cos(self._alpha))
        CD = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2 / (np.pi * MAV.e * MAV.AR)

        # compute Lift and Drag Forces (F_lift, F_drag)
        temp = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing
        F_lift = temp * (CL + MAV.C_L_q * MAV.c / (2 * self._Va) * q + MAV.C_L_delta_e * delta.elevator)
        F_drag = temp * (CD + MAV.C_D_q * MAV.c / (2 * self._Va) * q + MAV.C_D_delta_e * delta.elevator)

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

        # compute longitudinal forces in body frame (fx, fz)
        pg51_simplification = np.array([[np.cos(self._alpha), -np.sin(self._alpha)], [np.sin(self._alpha), np.cos(self._alpha)]]) @ np.array([[-F_drag], [-F_lift]])
        forces_xz = fg[::2] + pg51_simplification + np.array([[thrust_prop], [0]])
        fx = forces_xz.item(0)
        fz = forces_xz.item(1)
        # compute lateral forces in body frame (fy)
        fy = fg.item(1) + temp * (MAV.C_Y_0 + MAV.C_Y_beta * self._beta + MAV.C_Y_p * MAV.b / (2 * self._Va) * p + MAV.C_Y_r * MAV.b / (2 * self._Va) * r + MAV.C_Y_delta_a * delta.aileron + MAV.C_Y_delta_r * delta.rudder)
        # compute logitudinal torque in body frame (My)
        My = temp * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha * self._alpha + MAV.C_m_q * MAV.c / (2 * self._Va) * q + MAV.C_m_delta_e * delta.elevator)
        # compute lateral torques in body frame (Mx, Mz)
        Mx = temp * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta * self._beta + MAV.C_ell_p * MAV.b / (2 * self._Va) * p + MAV.C_ell_r * MAV.b / (2 * self._Va) * r + MAV.C_ell_delta_a * delta.aileron + MAV.C_ell_delta_r * delta.rudder) - torque_prop
        Mz = temp * MAV.b * (MAV.C_n_0 + MAV.C_n_beta * self._beta + MAV.C_n_p * MAV.b / (2 * self._Va) * p + MAV.C_n_r * MAV.b / (2 * self._Va) * r + MAV.C_n_delta_a * delta.aileron + MAV.C_n_delta_r * delta.rudder)

        # forces_moments = np.array([[0, 0, 0, 0, 0, 0]]).T
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = MAV.V_max * delta_t

        # Angular speed of propeller (omega_p = ?)
        a = ((MAV.rho * MAV.D_prop**5) / ((2 * np.pi)**2)) * MAV.C_Q0
        b = ((MAV.rho * MAV.D_prop**4) / (2 * np.pi)) * MAV.C_Q1 * Va + ((MAV.KQ * MAV.KV) / MAV.R_motor)
        c = MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2 - (MAV.KQ / MAV.R_motor) * v_in + MAV.KQ * MAV.i0
        omega_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        
        # Advance ratio
        J_op = (2 * np.pi * Va) / (omega_p * MAV.D_prop)
        # Coefficient of thrust and torque
        C_T = MAV.C_T2 * J_op**2 + MAV.C_T1 * J_op + MAV.C_T0
        C_Q = MAV.C_Q2 * J_op**2 + MAV.C_Q1 * J_op + MAV.C_Q0
        
        # thrust and torque due to propeller
        thrust_prop = ((MAV.rho * MAV.D_prop**4) / (4 * np.pi**2)) * omega_p**2 * C_T
        torque_prop = ((MAV.rho * MAV.D_prop**5) / (4 * np.pi**2)) * omega_p**2 * C_Q

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
        
        # print(self.true_state.north)
        # print(self.true_state.east)
        # print(self.true_state.altitude)
        # print(self.true_state.Va)
        # print(self.true_state.alpha)
        # print(self.true_state.beta)
        # print(self.true_state.phi)
        # print(self.true_state.theta)
        # print(self.true_state.psi)
        # print(self.true_state.Vg)
        # print(self.true_state.gamma)
        # print(self.true_state.chi)
        # print(self.true_state.p)
        # print(self.true_state.q)
        # print(self.true_state.r)
        # print(self.true_state.wn)
        # print(self.true_state.we)
        # self.true_state.bx = 0
        # self.true_state.by = 0
        # self.true_state.bz = 0
        # self.true_state.camera_az = 0
        # self.true_state.camera_el = 0
