"""
point_gimbal
    - point gimbal at target
part of mavsim
    - Beard & McLain, PUP, 2012
    - Update history:  
        3/31/2022 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from tools.rotations import Euler2Rotation
import parameters.camera_parameters as CAM


class Gimbal:
    def pointAtGround(self, mav):
        ###### TODO #######
        # desired inertial frame vector points down
        
        # rotate line-of-sight vector into body frame and normalize
        
        ell = np.array([[0],[0],[0]])
        return( self.pointAlongVector(ell, mav.camera_az, mav.camera_el) )

    def pointAtPosition(self, mav, target_position):
        ###### TODO #######
        # line-of-sight vector in the inertial frame
        p_mav = np.array([[mav.north], [mav.east], [-mav.altitude]])
        
        ell_inertial_desired = target_position - p_mav
        
        # rotate line-of-sight vector into body frame and normalize
        ell = (Euler2Rotation(mav.phi, mav.theta, mav.psi).T @ ell_inertial_desired) / np.linalg.norm(ell_inertial_desired)
        return( self.pointAlongVector(ell, mav.camera_az, mav.camera_el) )

    def pointAlongVector(self, ell, azimuth, elevation):
        # point gimbal so that optical axis aligns with unit vector ell
        # ell is assumed to be aligned in the body frame
        # given current azimuth and elevation angles of the gimbal

        ##### TODO #####
        # compute control inputs to align gimbal
        alpha_az_commanded = np.arctan2(ell.item(1), ell.item(0))
        alpha_el_commanded = -np.arcsin(ell.item(2))
        # alpha_el_commanded = 1
        
        
        # print('alpha_az_commanded = ', np.degrees(alpha_az_commanded))
        # print('az_commanded = ', alpha_az_commanded)
        # print('azimuth = ', np.degrees(azimuth))
        # print('alpha_el_commanded = ', np.degrees(alpha_el_commanded))
        # print('el_commanded = ', alpha_el_commanded)
        # print('elevation = ', np.degrees(elevation))
        # print('az_err', alpha_az_commanded - azimuth)
        # print('el_err', alpha_el_commanded - elevation)
        
        # proportional control for gimbal
        u_az = CAM.k_az * (alpha_az_commanded - azimuth)
        u_el = CAM.k_el * (alpha_el_commanded - elevation)
        return( np.array([[u_az], [u_el]]) )




