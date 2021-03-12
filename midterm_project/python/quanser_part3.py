import matplotlib.pyplot as plt
import numpy as np
from common import *
from direct_linear_transform import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('data/K.txt')
        # self.heli_points = np.loadtxt('data/heli_points.txt').T
        self.XY = np.loadtxt("data/platform_corners_metric.txt")
        # self.XY = np.loadtxt("data/platform_corners_metric_t23.txt") # Task 2.3
        self.platform_to_camera = np.loadtxt('data/platform_to_camera.txt')

    def residuals(self, uv, weights, lengths, markers, angles):
        # Extract input data
        l_base =            lengths[0]  # 0.1145
        l_hinge =           lengths[1]  # 0.325
        l_hinge_normal =    lengths[2]  # 0.050
        l_arm =             lengths[3]  # 0.65
        l_rotor_norm =      lengths[4]  # 0.030
        self.heli_points = markers
        yaw, pitch, roll = angles[:,0], angles[:,1], angles[:,2]

        # Compute the helicopter coordinate frames
        base_to_platform = translate(l_base/2, l_base/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  l_hinge)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -l_hinge_normal)
        rotors_to_arm    = translate(l_arm, 0.00, -l_rotor_norm)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        marker_locations = np.array([np.hstack([p1[i], p2[i]]) for i in range(p1.shape[0])])
        uv_hat = project(self.K, marker_locations)
        self.uv_hat = uv_hat # Save for use in draw()

        # Compute the vector of residuals.
        # r = np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]])
        r = np.array([np.hstack([(uv_hat[i,0]-uv[0])*weights[None,i,:], (uv_hat[i,1]-uv[1])*weights[None,i,:]]) for i in range(uv_hat.shape[0])])
        r = r.reshape(r.shape[0], r.shape[2])
        # r.sum(axis=0) # ?

        return r

    def residuals2(self, uv, R0, p):
        yaw, pitch, roll = p[0], p[1], p[2]
        t1, t2, t3 = p[3], p[4], p[5]

        # Compute the helicopter coordinate frames
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -0.050)
        rotors_to_arm    = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # u_tilde = np.append(uv, np.ones(4)).reshape((3,4))
        # xy_tilde = np.linalg.inv(self.K) @ u_tilde
        # xy = np.array([xy_tilde[0]/xy_tilde[2], xy_tilde[1]/xy_tilde[2]])
        # H = estimate_H(xy, self.XY)

        # T = decompose_H(H)
        T = np.zeros([4,4])
        R = rotate_x(yaw)[:3,:3] @ rotate_y(pitch)[:3,:3] @ rotate_z(roll)[:3,:3] @ R0
        T[:3,:3] = R
        T[:3, 3] = np.array([t1, t2, t3])
        
        uv_tilde = self.K @ T[np.array([True, True, True, False])] @ self.XY
        self.uv_hat = np.array([uv_tilde[0]/uv_tilde[2], uv_tilde[1]/uv_tilde[2]])

        # the sum of squared reprojection errors
        r = np.hstack([(self.uv_hat[0]-uv[0]), (self.uv_hat[1]-uv[1])])
        # r = ((uv-uv_hat)**2)
        self.T = T

        return r

    def get_optimized_T(self):
        return self.T

    def draw(self, uv, weights, image_number):
        I = plt.imread('quanser_sequence/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*uv[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.uv_hat, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
        plt.savefig('out_reprojection.png')
