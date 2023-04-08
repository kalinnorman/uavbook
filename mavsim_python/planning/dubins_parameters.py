# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB
#         4/2/2020 - RWB
#         3/30/2022 - RWB

import numpy as np
import sys
sys.path.append('..')


class DubinsParameters:

    def update(self, ps, chis, pe, chie, R):
        self.p_s = ps
        self.chi_s = chis
        self.p_e = pe
        self.chi_e = chie
        self.radius = R
        self.compute_parameters()

    def compute_parameters(self):
        ps = self.p_s
        pe = self.p_e
        chis = self.chi_s
        chie = self.chi_e
        R = self.radius
        ell = np.linalg.norm(ps[0:2] - pe[0:2])

        ##### TODO #####
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
        else:
            # compute start and end circles
            crs = self.p_s + R * rotz(np.pi / 2) @ np.array([[np.cos(chis), np.sin(chis), 0]]).T
            cls = self.p_s + R * rotz(-np.pi / 2) @ np.array([[np.cos(chis), np.sin(chis), 0]]).T
            cre = self.p_e + R * rotz(np.pi / 2) @ np.array([[np.cos(chie), np.sin(chie), 0]]).T
            cle = self.p_e + R * rotz(-np.pi / 2) @ np.array([[np.cos(chie), np.sin(chie), 0]]).T

            origin_vec = np.array([[1, 0, 0]]).T
            # compute L1
            line = cre - crs
            v = np.arccos((line.T @ origin_vec) / (np.linalg.norm(line) * np.linalg.norm(origin_vec)))
            L1 = np.linalg.norm(crs - cre) + \
                 R * mod(2 * np.pi + mod(v - np.pi / 2) - mod(chis - np.pi / 2)) + \
                 R * mod(2 * np.pi + mod(chie - np.pi / 2) - mod(v - np.pi / 2))

            # compute L2
            line = cle - crs
            v = np.arccos((line.T @ origin_vec) / (np.linalg.norm(line) * np.linalg.norm(origin_vec)))
            v2 = v - np.pi / 2 + np.arcsin((2 * R) / np.linalg.norm(cle - crs))
            L2 = np.sqrt(np.linalg.norm(cle - crs)**2 - 4 * R**2) + \
                 R * mod(2 * np.pi + mod(v2) - mod(chis - np.pi / 2)) + \
                 R * mod(2 * np.pi + mod(v2 + np.pi) - mod(chie + np.pi / 2))

            # compute L3
            line = cre - cls
            v = np.arccos((line.T @ origin_vec) / (np.linalg.norm(line) * np.linalg.norm(origin_vec)))
            v2 = np.arccos((2 * R) / np.linalg.norm(line))
            L3 = np.sqrt(np.linalg.norm(line)**2 - 4 * R**2) + \
                 R * mod(2 * np.pi + mod(chis + np.pi / 2) - mod(v + v2)) + \
                 R * mod(2 * np.pi + mod(chie - np.pi / 2) - mod(v + v2 - np.pi))

            # compute L4
            line = cle - cls
            v = np.arccos((line.T @ origin_vec) / (np.linalg.norm(line) * np.linalg.norm(origin_vec)))
            L4 = np.linalg.norm(cls - cle) + \
                 R * mod(2 * np.pi + mod(chis + np.pi / 2) - mod(v + np.pi / 2)) + \
                 R * mod(2 * np.pi + mod(v + np.pi / 2) - mod(chie + np.pi / 2))

            # L is the minimum distance
            L = np.min([L1, L2, L3, L4])
            min_idx = np.argmin([L1, L2, L3, L4])

            if min_idx == 0:
                cs = crs
                dirs = 1
                ce = cre
                dire = 1
                q1 = (ce - cs) / np.linalg.norm(ce - cs)
                z1 = cs + R * rotz(-np.pi / 2) @ q1
                z2 = ce + R * rotz(-np.pi / 2) @ q1
            elif min_idx == 1:
                cs = crs
                dirs = 1
                ce = cle
                dire = -1
                elll = np.linalg.norm(ce - cs)
                v = np.arccos(((ce - cs).T @ origin_vec) / (np.linalg.norm(ce - cs) * np.linalg.norm(origin_vec)))
                v2 = v - np.pi / 2 + np.arcsin((2 * R) / elll)
                q1 = rotz(v2 + np.pi/2) @ origin_vec
                z1 = cs + R * rotz(v2) @ origin_vec
                z2 = ce + R * rotz(v2 + np.pi) @ origin_vec
            elif min_idx == 2:
                cs = cls
                dirs = -1
                ce = cre
                dire = 1
                elll = np.linalg.norm(ce - cs)
                v = np.arccos(((ce - cs).T @ origin_vec) / (np.linalg.norm(ce - cs) * np.linalg.norm(origin_vec)))
                v2 = np.arccos((2 * R) / elll)
                q1 = rotz(v + v2 - np.pi / 2) @ origin_vec
                z1 = cs + R * rotz(v + v2) @ origin_vec
                z2 = ce + R * rotz(v + v2 - np.pi) @ origin_vec
            elif min_idx == 3:
                cs = cls
                dirs = -1
                ce = cle
                dire = -1
                q1 = (ce - cs) / np.linalg.norm(ce - cs)
                z1 = cs + R * rotz(np.pi / 2) @ q1
                z2 = ce + R * rotz(np.pi / 2) @ q1
            z3 = pe
            q2 = rotz(chie) @ origin_vec 
            
            self.length = L
            self.center_s = cs
            self.dir_s = dirs
            self.center_e = ce
            self.dir_e = dire
            self.r1 = z1
            self.n1 = q1
            self.r2 = z2
            self.r3 = z3
            self.n3 = q2

    def compute_points(self):
        ##### TODO ##### - uncomment lines and remove last line
        Del = 0.1  # distance between point

        # points along start circle
        th1 = np.arctan2(self.p_s.item(1) - self.center_s.item(1),
                         self.p_s.item(0) - self.center_s.item(0))
        th1 = mod(th1)
        th2 = np.arctan2(self.r1.item(1) - self.center_s.item(1),
                         self.r1.item(0) - self.center_s.item(0))
        th2 = mod(th2)
        th = th1
        theta_list = [th]
        if self.dir_s > 0:
            if th1 >= th2:
                while th < th2 + 2*np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2*np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)

        points = np.array([[self.center_s.item(0) + self.radius * np.cos(theta_list[0]),
                            self.center_s.item(1) + self.radius * np.sin(theta_list[0]),
                            self.center_s.item(2)]])
        for angle in theta_list:
            new_point = np.array([[self.center_s.item(0) + self.radius * np.cos(angle),
                                   self.center_s.item(1) + self.radius * np.sin(angle),
                                   self.center_s.item(2)]])
            points = np.concatenate((points, new_point), axis=0)

        # points along straight line
        sig = 0
        while sig <= 1:
            new_point = np.array([[(1 - sig) * self.r1.item(0) + sig * self.r2.item(0),
                                   (1 - sig) * self.r1.item(1) + sig * self.r2.item(1),
                                   (1 - sig) * self.r1.item(2) + sig * self.r2.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
            sig += Del

        # points along end circle
        th2 = np.arctan2(self.p_e.item(1) - self.center_e.item(1),
                         self.p_e.item(0) - self.center_e.item(0))
        th2 = mod(th2)
        th1 = np.arctan2(self.r2.item(1) - self.center_e.item(1),
                         self.r2.item(0) - self.center_e.item(0))
        th1 = mod(th1)
        th = th1
        theta_list = [th]
        if self.dir_e > 0:
            if th1 >= th2:
                while th < th2 + 2 * np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2 * np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)
        for angle in theta_list:
            new_point = np.array([[self.center_e.item(0) + self.radius * np.cos(angle),
                                   self.center_e.item(1) + self.radius * np.sin(angle),
                                   self.center_e.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
        # points = np.zeros((5,3))
        return points


def rotz(theta):
    if isinstance(theta, np.ndarray):
        while isinstance(theta, np.ndarray):
            theta = theta.item(0)
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def mod(x):
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


