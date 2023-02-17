"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Quaternion2Euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input: MsgDelta):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])

    ###### TODO ######
    # define transfer function constants
    # See equations 5.23 and 5.24 in the text (pg 75) -- For Va do I put in Va_trim??
    a_phi1 = -0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_p * MAV.b / (2 * Va_trim)
    a_phi2 = 0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a
    # See above equation 5.29 in the text (pg 79)
    a_theta1 = -0.5 * MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_q * MAV.c / (2 * MAV.Jy * Va_trim)
    a_theta2 = -0.5 * MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_alpha / MAV.Jy
    a_theta3 = 0.5 * MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_delta_e / MAV.Jy

    # Compute transfer function coefficients using new propulsion model
    ## See below equation 5.36 in the text (pg 83)
    a_V1 = MAV.rho * Va_trim * MAV.S_wing / MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e * trim_input.elevator) - 1 / MAV.mass * dT_dVa(mav, Va_trim, trim_input.throttle)
    a_V2 = 1 / MAV.mass * dT_ddelta_t(mav, Va_trim, trim_input.throttle)
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    
    ##### TODO #####
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # extract longitudinal states (u, w, q, theta, pd)
    E1 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    E2 = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 1]])
    A_lon = E1 @ A @ E1.T
    B_lon = E1 @ B @ E2.T
    # change pd to h

    # extract lateral states (v, p, r, phi, psi)
    E3 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    E4 = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0]])
    A_lat = E3 @ A @ E3.T
    B_lat = E3 @ B @ E4.T
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    
    x_euler = np.zeros((12,1))
    x_euler[0:6] = np.copy(x_quat[0:6])
    x_euler[6:9, 0] = Quaternion2Euler(np.copy(x_quat[6:10]))
    x_euler[9:12] = np.copy(x_quat[10:13])
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions

    x_quat = np.zeros((13,1))
    x_quat[0:6] = np.copy(x_euler[0:6])
    x_quat[6:10] = Euler2Quaternion(*x_euler[6:9].squeeze())
    x_quat[10:13] = np.copy(x_euler[9:12])
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of Quaternion2Euler(quat) with respect to quat
    # compute partial Quaternion2Euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    ##### TODO #####
    if isinstance(delta, np.ndarray):
        delta = MsgDelta(delta.item(0), delta.item(1), delta.item(2), delta.item(3))
    dquat_dt = mav._derivatives(x_quat, mav._forces_moments(delta))
    f_euler_ = np.zeros((12,1))
    temp = np.zeros((12, 13))
    eps = 0.001
    f_at_x = euler_state(x_quat)
    for i in range(13):
        eps_state = np.copy(mav._state)
        eps_state[i, 0] += eps
        f_at_eps = euler_state(eps_state)
        df_dq = (f_at_eps - f_at_x) / eps
        temp[:, i] = df_dq[:, 0]
    f_euler_ = temp @ dquat_dt
    return f_euler_

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.01  # deviation

    A = np.zeros((12, 12))  # Jacobian of f wrt x
    # See slide 48 of chap 5
    f_at_x = f_euler(mav, x_euler, delta)
    for i in range(0, 12):
        x_eps = np.copy(x_euler)
        x_eps[i][0] += eps # add eps to i th s ta te
        f_at_x_eps = f_euler(mav, x_eps, delta)
        df_dxi = (f_at_x_eps - f_at_x ) / eps
        A[:, i] = df_dxi[:, 0]
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.01  # deviation

    B = np.zeros((12, 4))  # Jacobian of f wrt u
    delta = delta.to_array()[:4, :]
    # See slide 48 of chap 5
    f_at_u = f_euler(mav, x_euler, delta)
    for i in range(0, 4):
        u_eps = np.copy(delta)
        u_eps[i][0] += eps # add eps to i th s ta te
        f_at_u_eps = f_euler(mav, x_euler, u_eps)
        df_dui = (f_at_u_eps - f_at_u) / eps
        B[:, i] = df_dui[:, 0]
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01

    dT_dVa = 0
    T_at_Va = mav._motor_thrust_torque(Va, delta_t)[0]
    Va_eps = Va + eps
    T_at_Va_eps = mav._motor_thrust_torque(Va_eps, delta_t)[0]
    dT_dVa = (T_at_Va_eps - T_at_Va) / eps
    return dT_dVa

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01

    dT_ddelta_t = 0
    T_at_delta_t = mav._motor_thrust_torque(Va, delta_t)[0]
    delta_t_eps = delta_t + eps
    T_at_delta_t_eps = mav._motor_thrust_torque(Va, delta_t_eps)[0]
    dT_ddelta_t = (T_at_delta_t_eps - T_at_delta_t) / eps
    return dT_ddelta_t
