"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - RWB
        3/30/2022 - RWB
"""

import numpy as np
import sys
sys.path.append('..')
from planning.dubins_parameters import DubinsParameters
from message_types.msg_path import MsgPath


class PathManager:
    def __init__(self):
        # message sent to path follower
        self.path = MsgPath()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        self.manager_requests_waypoints = True
        self.dubins_path = DubinsParameters()

    def update(self, waypoints, radius, state):
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False
        if waypoints.type == 'straight_line':
            self.line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self.fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            self.dubins_manager(waypoints, radius, state)
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        if waypoints.flag_waypoints_changed:
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            self.path.type = 'line'
            self.path.line_origin = waypoints.ned[:, self.ptr_previous].reshape((3, 1))
            pwpt = waypoints.ned[:, self.ptr_previous].reshape((3, 1))
            cwpt = waypoints.ned[:, self.ptr_current].reshape((3, 1))
            qprev = (cwpt - pwpt) / np.linalg.norm(cwpt - pwpt)
            self.path.line_direction = qprev
            self.path.plot_updated = False
        pwpt = waypoints.ned[:, self.ptr_previous].reshape((3, 1))
        cwpt = waypoints.ned[:, self.ptr_current].reshape((3, 1))
        nwpt = waypoints.ned[:, self.ptr_next].reshape((3, 1))
        qprev = (cwpt - pwpt) / np.linalg.norm(cwpt - pwpt)
        qcurr = (nwpt - cwpt) / np.linalg.norm(nwpt - cwpt)
        ncurr = (qprev + qcurr) / np.linalg.norm(qprev + qcurr)
        # Check if we are past the half plane
        if (mav_pos - cwpt).T @ ncurr >= 0:
            if self.ptr_next < waypoints.num_waypoints - 1:
                self.increment_pointers()
                self.path.type = 'line'
                self.path.line_origin = cwpt
                self.path.line_direction = qcurr
                self.path.plot_updated = False
            else:
                self.path.type = 'line'
                self.path.line_origin = cwpt
                self.path.line_direction = qcurr
                self.path.plot_updated = False

    def fillet_manager(self, waypoints, radius, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        if waypoints.flag_waypoints_changed:
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            self.path.type = 'line'
            self.path.line_origin = waypoints.ned[:, self.ptr_previous].reshape((3, 1))
            pwpt = waypoints.ned[:, self.ptr_previous].reshape((3, 1))
            cwpt = waypoints.ned[:, self.ptr_current].reshape((3, 1))
            qprev = (cwpt - pwpt) / np.linalg.norm(cwpt - pwpt)
            self.path.line_direction = qprev
            self.path.plot_updated = False
        pwpt = waypoints.ned[:, self.ptr_previous].reshape((3, 1))
        cwpt = waypoints.ned[:, self.ptr_current].reshape((3, 1))
        nwpt = waypoints.ned[:, self.ptr_next].reshape((3, 1))
        qprev = (cwpt - pwpt) / np.linalg.norm(cwpt - pwpt)
        qcurr = (nwpt - cwpt) / np.linalg.norm(nwpt - cwpt)
        qangle = np.arccos(-qprev.T @ qcurr)
        if self.manager_state == 1:
            z = cwpt - (radius / np.tan(qangle / 2)) * qprev
            # check if we are past the half plane
            if (mav_pos - z).T @ qprev >= 0:
                self.manager_state = 2
                self.path.type = 'orbit'
                self.path.orbit_center = cwpt - (radius / np.sin(qangle / 2)) * ((qprev - qcurr) / np.linalg.norm(qprev - qcurr))
                self.path.orbit_radius = radius
                orbit_lambda = np.sign(qprev[0, 0] * qcurr[1, 0] - qprev[1, 0] * qcurr[0, 0])
                if orbit_lambda > 0: 
                    self.path.orbit_direction = 'CW'
                else:
                    self.path.orbit_direction = 'CCW'
                self.path.plot_updated = False
        elif self.manager_state == 2:
            z = cwpt + (radius / np.tan(qangle / 2)) * qcurr
            if (mav_pos - z).T @ qcurr >= 0:
                self.manager_state = 1
                if self.ptr_next < waypoints.num_waypoints - 1:
                    self.increment_pointers()
                    self.path.type = 'line'
                    self.path.line_origin = cwpt
                    self.path.line_direction = qcurr
                    self.path.plot_updated = False
                else:
                    self.path.type = 'line'
                    self.path.line_origin = cwpt
                    self.path.line_direction = qcurr
                    self.path.plot_updated = False
      

    def dubins_manager(self, waypoints, radius, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        if waypoints.flag_waypoints_changed:
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            self.dubins_path.update(mav_pos, state.chi, waypoints.ned[:, self.ptr_current].reshape((3, 1)), waypoints.course[self.ptr_current], radius)
            self.starting = True
        if self.manager_state == 1:
            self.path.type = 'orbit'
            self.path.orbit_center = self.dubins_path.center_s
            self.path.orbit_radius = radius
            self.path.plot_updated = False
            if self.dubins_path.dir_s > 0: 
                self.path.orbit_direction = 'CW'
            else:
                self.path.orbit_direction = 'CCW'
            if (mav_pos - self.dubins_path.r1).T @ -self.dubins_path.n1 >= 0 or self.starting == True:
                self.manager_state = 2
        elif self.manager_state == 2:
            if (mav_pos - self.dubins_path.r1).T @ self.dubins_path.n1 >= 0 or self.starting == True:
                self.manager_state = 3
                self.starting = False
        elif self.manager_state == 3:
            self.path.type = 'line'
            self.path.line_origin = self.dubins_path.r1
            self.path.line_direction = self.dubins_path.n1
            self.path.plot_updated = False
            if (mav_pos - self.dubins_path.r2).T @ self.dubins_path.n1 >= 0:
                self.manager_state = 4
        elif self.manager_state == 4:
            self.path.type = 'orbit'
            self.path.orbit_center = self.dubins_path.center_e
            self.path.orbit_radius = radius
            self.path.plot_updated = False
            if self.dubins_path.dir_e > 0: 
                self.path.orbit_direction = 'CW'
            else:
                self.path.orbit_direction = 'CCW'
            if (mav_pos - self.dubins_path.r3).T @ -self.dubins_path.n3 >= 0:
                self.manager_state = 5
        elif self.manager_state == 5:
            if (mav_pos - self.dubins_path.r3).T @ self.dubins_path.n3 >= 0:
                self.manager_state = 1
                self.increment_pointers()
                self.dubins_path.update(mav_pos, state.chi, waypoints.ned[:, self.ptr_current].reshape((3, 1)), waypoints.course[self.ptr_current], radius)
        ##### TODO #####
        # Use functions - self.initialize_pointers(), self.dubins_path.update(),
        # self.construct_dubins_circle_start(), self.construct_dubins_line(),
        # self.inHalfSpace(), self.construct_dubins_circle_end(), self.increment_pointers(),

        # Use variables - self.num_waypoints, self.dubins_path, self.ptr_current,
        # self.ptr_previous, self.manager_state, self.manager_requests_waypoints,
        # waypoints.__, radius


    def initialize_pointers(self):
        if self.num_waypoints >= 3:
            ##### TODO #####
            self.ptr_previous = 0
            self.ptr_current = 1
            self.ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def increment_pointers(self):
        ##### TODO #####
        self.ptr_previous += 1
        self.ptr_current += 1
        self.ptr_next += 1

    # def construct_line(self, waypoints):
    #     previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous+1]
    #     ##### TODO #####
    #     # current = ?
    #     # next = ?
       
    #     # update halfspace variables
    #     # self.halfspace_n =
    #     # self.halfspace_r = 
        
    #     # Update path variables
    #     # self.path.__ =

    # def construct_fillet_line(self, waypoints, radius):
    #     previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous+1]
    #     ##### TODO #####
    #     # current = ?
    #     # next = ?

    #     # update halfspace variables
    #     # self.halfspace_n =
    #     # self.halfspace_r = 
        
    #     # Update path variables
    #     # self.path.__ =

    # def construct_fillet_circle(self, waypoints, radius):
    #     previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous+1]
    #     ##### TODO #####
    #     # current = ?
    #     # next = ?

    #     # update halfspace variables
    #     # self.halfspace_n =
    #     # self.halfspace_r = 
        
    #     # Update path variables
    #     # self.path.__ =

    # def construct_dubins_circle_start(self, waypoints, dubins_path):
    #     ##### TODO #####
    #     # update halfspace variables
    #     # self.halfspace_n =
    #     # self.halfspace_r = 
        
    #     # Update path variables
    #     # self.path.__ =
    #     pass

    # def construct_dubins_line(self, waypoints, dubins_path):
    #     ##### TODO #####
    #     # update halfspace variables
    #     # self.halfspace_n =
    #     # self.halfspace_r = 
        
    #     # Update path variables
    #     # self.path.__ =
    #     pass

    # def construct_dubins_circle_end(self, waypoints, dubins_path):
    #     ##### TODO #####
    #     # update halfspace variables
    #     # self.halfspace_n =
    #     # self.halfspace_r = 
        
    #     # Update path variables
    #     # self.path.__ =
    #     pass

    # def inHalfSpace(self, pos):
    #     if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
    #         return True
    #     else:
    #         return False

