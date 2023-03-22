import numpy as np
x_trim = np.array([[0.000000, 0.000000, -100.000000, 24.968623, 0.000000, 1.252151, 0.999686, 0.000000, 0.025051, 0.000000, 0.000000, 0.000000, 0.000000]]).T
u_trim = np.array([[-0.125044, 0.001837, -0.000303, 0.676775]]).T
Va_trim = 25.000000
alpha_trim = 0.050107
theta_trim = 0.050107
a_phi1 = 22.628851
a_phi2 = 130.883678
a_theta1 = 5.294738
a_theta2 = 99.947422
a_theta3 = -36.112390
a_V1 = 0.055242
a_V2 = 8.207536
a_V3 = 9.810000
A_lon = np.array([
    [-0.206751, 0.501088, -1.222177, -9.795068, -0.000000],
    [-0.561011, -4.463973, 24.370933, -0.540322, -0.000000],
    [0.200319, -3.992960, -5.294738, 0.000000, -0.000000],
    [0.000000, 0.000000, 0.999974, 0.000000, -0.000000],
    [0.050086, -0.998745, -0.000000, 24.999583, 0.000000]])
B_lon = np.array([
    [-0.138152, 8.207536],
    [-2.586197, 0.000000],
    [-36.112390, 0.000000],
    [0.000000, 0.000000],
    [-0.000000, -0.000000]])
A_lat = np.array([
    [-0.776773, 1.252151, -24.968623, 9.797524, 0.000000],
    [-3.866719, -22.628851, 10.905041, 0.000000, 0.000000],
    [0.783077, -0.115092, -1.227655, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.050149, 0.000000, 0.000000],
    [0.000000, -0.000000, 1.001256, 0.000000, 0.000000]])
B_lat = np.array([
    [1.486172, 3.764969],
    [130.883678, -1.796374],
    [5.011735, -24.881341],
    [0.000000, 0.000000],
    [0.000000, 0.000000]])
Ts = 0.010000