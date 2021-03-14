import matplotlib.pyplot as plt
import numpy as np
from common import *

class Quanser2:
    def __init__(self):
        self.K = np.loadtxt('midterm_project/data/K.txt')
        #self.heli_points = np.loadtxt('midterm_project/data/heli_points.txt').T
        self.heli_points = np.array([[-0.13116262,  0.17766911,  0.43327253, -0.03368243, -0.03428176, -0.02950639, -0.03133537],
         [-0.0092462 , -0.00968803, -0.01006848, -0.08953217, -0.17767401, 0.21100523,  0.10739365],
         [ 0.00647913,  0.01080793,  0.01314459, -0.03126685, -0.05092014, -0.0351117 , -0.03136749],
         [1, 1, 1, 1, 1, 1, 1]])
        self.platform_to_camera = np.loadtxt('midterm_project/data/platform_to_camera.txt')
        self.lengths = np.array([0.112, 0.321, 0.051, 0.661, 0.031])

    def calculate_uv_hat(self, p, angles, K):
        base_to_platform = rotate_x(p[28]) @ rotate_y(p[29]) @ translate(p[0]/2,p[0]/2, 0.0) @ rotate_z(angles[0])
        hinge_to_base = rotate_x(p[30]) @ rotate_z(p[31]) @ translate(p[5], 0.00,  p[1]) @ rotate_y(angles[1])
        arm_to_hinge = translate(0.00, 0.00, -p[2])
        rotors_to_arm = rotate_y(p[32]) @ rotate_z(p[33]) @ translate(p[3], p[6], -p[4])  @ rotate_x(angles[2])
        base_to_camera   = self.platform_to_camera @ base_to_platform
        hinge_to_camera  = base_to_camera @ hinge_to_base
        arm_to_camera    = hinge_to_camera @ arm_to_hinge
        rotors_to_camera = arm_to_camera @ rotors_to_arm

        # Compute the predicted image location of the markers
        markers_arm = np.vstack((np.reshape(p[7:16], (3,3)).T, np.ones(3)))
        markers_rotor = np.vstack((np.reshape(p[16:28], (4,3)).T, np.ones(4)))
        p1 = arm_to_camera @ markers_arm
        p2 = rotors_to_camera @ markers_rotor
        uv_hat = project(K, np.hstack([p1, p2]))

        return uv_hat
    

    def residuals3(self, uv, weights, p, K, n, m, l):
        angles = p[m:]
        angles = angles.reshape((angles.shape[0]//3,3))
        rs = np.array([])
        
        for j in range(angles.shape[0]):
            uv_hat = self.calculate_uv_hat(p[:m], angles[j], K)
            r = np.zeros(2*n)

            for i in range(n):
                r[i] = (uv_hat[0][i] - uv[j][i]) * weights[j][i]
                r[n+i] = (uv_hat[1][i] - uv[l+j][i]) * weights[j][i]
            rs = np.append(rs, r)
        
        return rs

    def residuals4(self, uv, weights, p, K, image_number, n, m, l):
        angles = p[m:]
        angles = angles.reshape((angles.shape[0]//3,3))
        rs = np.array([])
        
        for j in range(1):
            uv_hat = self.calculate_uv_hat(p[:m], angles[j], K)
            r = np.zeros(2*n)

            for i in range(n):
                r[i] = (uv_hat[0][i] - uv[image_number][i]) * weights[image_number][i]
                r[n+i] = (uv_hat[1][i] - uv[l+image_number][i]) * weights[image_number][i]
            rs = np.append(rs, r)
        
        return rs

        def draw(self, uv, weights, image_number):
            I = plt.imread('../data/video%04d.jpg' % image_number)
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
