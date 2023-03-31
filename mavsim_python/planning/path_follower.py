import numpy as np
from math import sin, cos
import sys

sys.path.append('..')
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap


class PathFollower:
    def __init__(self):
        ##### TODO #####
        self.chi_inf = np.pi / 3  # approach angle for large distance from straight-line path
        self.k_path = 0.07  # path gain for straight-line path following
        self.k_orbit = 5 # path gain for orbit following
        self.gravity = 9.81
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        rn, re, rd = path.line_origin.flatten()
        qn, qe, qd = path.line_direction.flatten()
        chiq = np.arctan2(qe, qn)
        while chiq - state.chi < -np.pi:
            chiq += 2 * np.pi
        while chiq - state.chi > np.pi:
            chiq -= 2 * np.pi

        #airspeed command
        self.autopilot_commands.airspeed_command = 25

        # altitude command
        R_i_to_P = np.array([[cos(chiq), sin(chiq), 0],
                             [-sin(chiq), cos(chiq), 0],
                             [0, 0, 1]])
        pi = np.array([[state.north, state.east, -state.altitude]]).T
        ep = R_i_to_P @ (pi - path.line_origin)
        n = np.cross(np.array([0, 0, 1]), path.line_direction.flatten()) / np.linalg.norm(np.cross(np.array([0, 0, 1]), path.line_direction.flatten()))
        n = n.reshape(3, 1)
        si = ep - ep.T @ n * n
        
        self.autopilot_commands.altitude_command = -rd - np.sqrt(si.item(0)**2 + si.item(1)**2) * (qd / np.sqrt(qn**2 + qe**2))

        # course command
        self.autopilot_commands.course_command = chiq - self.chi_inf * (2 / np.pi) * np.arctan(self.k_path * ep.item(1))

        # feedforward roll angle for straight line is zero
        self.autopilot_commands.phi_feedforward = 0 

    def _follow_orbit(self, path, state):
        cn, ce, cd = path.orbit_center.flatten()
        radius = path.orbit_radius
        direction = path.orbit_direction
        pn, pe, pd = state.north, state.east, -state.altitude
        
        d = np.sqrt((pn - cn)**2 + (pe - ce)**2)
        Phi = np.arctan2(pe - ce, pn - cn)
        while Phi - state.chi < -np.pi:
            Phi += 2 * np.pi
        while Phi - state.chi > np.pi:
            Phi -= 2 * np.pi
            
        if direction == 'CW':
            lambda_val = 1
        else:
            lambda_val = -1
            
        # airspeed command
        self.autopilot_commands.airspeed_command = 25

        # course command
        self.autopilot_commands.course_command = Phi + lambda_val * (np.pi / 2 + np.arctan(self.k_orbit * (d - radius) / radius))

        # altitude command
        self.autopilot_commands.altitude_command = -cd
        
        # roll feedforward command
        other_phi = self.autopilot_commands.course_command # TODO : Fix this
        self.autopilot_commands.phi_feedforward = lambda_val * np.arctan2(state.Vg**2, self.gravity * radius * np.cos(state.chi - other_phi))




