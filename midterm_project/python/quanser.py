import matplotlib.pyplot as plt
import numpy as np
from common import *
from direct_linear_transform import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('data/K.txt')
        self.heli_points = np.loadtxt('data/heli_points.txt').T
        self.XY = np.loadtxt("data/platform_corners_metric.txt")
        # self.XY = np.loadtxt("data/platform_corners_metric_t23.txt") # Task 2.3
        self.platform_to_camera = np.loadtxt('data/platform_to_camera.txt')
        self.lengths = np.array([11.45, 32.5, 5, 65, 3])*1e-2

    def residuals(self, uv, weights, yaw, pitch, roll):
        # Compute the helicopter coordinate frames
        """
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -0.050)
        rotors_to_arm    = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm
        


        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()
        #
        # TASK: Compute the vector of residuals.
        #r = np.zeros(2*7) # Placeholder, remove me!
        #r = np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]])

        #return r[0]

        """
         # Compute the helicopter coordinate frames
        base_to_platform = translate(self.lengths[0]/2, self.lengths[0]/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  self.lengths[1])@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -self.lengths[2])
        rotors_to_arm    = translate(self.lengths[3], 0.00, -self.lengths[4])@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()

        #
        # TASK: Compute the vector of residuals.
        #
        n = uv.shape[1]
        r = np.zeros(2*n) # Placeholder, remove me!

        for i in range(n):
            r[i] = (uv_hat[0][i] - uv[0][i]) * weights[i]
            r[n+i] = (uv_hat[1][i] - uv[1][i]) * weights[i]

        return r

    def residuals2(self, uv, R0, p):
        yaw, pitch, roll = p[0], p[1], p[2]
        t1, t2, t3 = p[3], p[4], p[5]

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

    def residuals3(self, detections, p):
        # Compute the helicopter coordinate frames

        lengths = p[:5]
        markers = p[5:26]
        trajectory = p[26:]
        markers = np.vstack((markers.reshape(7,3).T,np.ones(7)))
        

        num_images = detections.shape[0]

        r = np.array([])
        #markers = markers.T

        for image_number in range(num_images):

            uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
            weights = detections[image_number, ::3]

            base_to_platform = translate(lengths[0]/2, lengths[0]/2, 0.0)@rotate_z(trajectory[image_number*3])
            hinge_to_base    = translate(0.00, 0.00,  lengths[1])@rotate_y(trajectory[image_number*3+1])
            arm_to_hinge     = translate(0.00, 0.00, -lengths[2])
            rotors_to_arm    = translate(lengths[3], 0.00, -lengths[4])@rotate_x(trajectory[image_number*3+2])
            base_to_camera   = self.platform_to_camera@base_to_platform
            hinge_to_camera  = base_to_camera@hinge_to_base
            arm_to_camera    = hinge_to_camera@arm_to_hinge
            rotors_to_camera = arm_to_camera@rotors_to_arm

            # Compute the predicted image location of the markers
            p1 = arm_to_camera @ markers[:,:3]
            p2 = rotors_to_camera @ markers[:,3:]
            uv_hat = project(self.K, np.hstack([p1, p2]))
            #uv_hat = uv_hat 
            # uv_hat = 2*7

            r = np.append(r, np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]]))

            """
            q = np.zeros(2*n)

            for i in range(n):
                q[i] = (uv_hat[0][i] - uv[0][i]) * weights[i]
                q[n+i] = (uv_hat[1][i] - uv[1][i]) * weights[i]
            r = np.append(r, q)
            """
            

        return r


    def residuals_individual(self, detections, p, image_number): # completing the right block. 3l x 2nl
        # Compute the helicopter coordinate frames

        lengths = p[:5]
        markers = p[5:26]
        trajectory = p[26:]
        markers = np.vstack((markers.reshape(7,3).T,np.ones(7)))

        r = np.array([])

        uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
        weights = detections[image_number, ::3]

        base_to_platform = translate(lengths[0]/2, lengths[0]/2, 0.0)@rotate_z(trajectory[0])
        hinge_to_base    = translate(0.00, 0.00,  lengths[1])@rotate_y(trajectory[1])
        arm_to_hinge     = translate(0.00, 0.00, -lengths[2])
        rotors_to_arm    = translate(lengths[3], 0.00, -lengths[4])@rotate_x(trajectory[2])
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ markers[:,:3]
        p2 = self.rotors_to_camera @ markers[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()
        # uv_hat = 2*7

        r = np.append(r, np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]]))
       
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

    def draw2(self, uv, uv_hat, weights, image_number):
        I = plt.imread('quanser_sequence/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*uv[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*uv_hat, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        """
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        """
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
        plt.savefig('out_reprojection_task2_imagenr%04d.png' % image_number)




class ImprovedQuanser:
    def __init__(self):
        self.K = np.loadtxt('data/K.txt')
        self.heli_points = np.loadtxt('data/improved_marker_points.txt').T
        self.XY = np.loadtxt("data/platform_corners_metric.txt")
        # self.XY = np.loadtxt("data/platform_corners_metric_t23.txt") # Task 2.3
        self.platform_to_camera = np.loadtxt('data/platform_to_camera.txt')
        self.lengths = np.array([0.11249211, 0.32140264, 0.05077297, 0.6611940541181118, 0.03120462792288483])

    def residuals(self, uv, weights, yaw, pitch, roll):
        # Compute the helicopter coordinate frames
        """
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -0.050)
        rotors_to_arm    = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm
        


        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()
        #
        # TASK: Compute the vector of residuals.
        #r = np.zeros(2*7) # Placeholder, remove me!
        #r = np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]])

        #return r[0]

        """
         # Compute the helicopter coordinate frames
        base_to_platform = translate(self.lengths[0]/2, self.lengths[0]/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  self.lengths[1])@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -self.lengths[2])
        rotors_to_arm    = translate(self.lengths[3], 0.00, -self.lengths[4])@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()

        #
        # TASK: Compute the vector of residuals.
        #
        n = uv.shape[1]
        r = np.zeros(2*n) # Placeholder, remove me!

        for i in range(n):
            r[i] = (uv_hat[0][i] - uv[0][i]) * weights[i]
            r[n+i] = (uv_hat[1][i] - uv[1][i]) * weights[i]

        return r

    

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

