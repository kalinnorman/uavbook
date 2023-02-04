"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
from tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state

        #   Dryden gust model parameters 
        from parameters.planner_parameters import Va0
        Lu = Lv = 200
        Lw = 50
        sigmau = sigmav = 1.06
        sigmaw = 0.7
        
        u_mult = sigmau * np.sqrt((2 * Va0) / (np.pi * Lu))
        v_mult = sigmav * np.sqrt((3 * Va0) / (np.pi * Lv))
        w_mult = sigmaw * np.sqrt((3 * Va0) / (np.pi * Lw))

        # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
        self.u_w = TransferFunction(num=np.array([[u_mult]]), 
                                    den=np.array([[1, Va0 / Lu]]),Ts=Ts)
        self.v_w = TransferFunction(num=np.array([[v_mult, v_mult * (Va0 / (np.sqrt(3) * Lv))]]), 
                                    den=np.array([[1, 2 * (Va0 / Lv), (Va0 / Lv)**2]]),Ts=Ts)
        self.w_w = TransferFunction(num=np.array([[w_mult, w_mult * (Va0 / (np.sqrt(3) * Lw))]]), 
                                    den=np.array([[1, 2 * (Va0 / Lw), (Va0 / Lw)**2]]),Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))

