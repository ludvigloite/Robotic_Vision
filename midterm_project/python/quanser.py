import matplotlib.pyplot as plt
import numpy as np
from common import *
from direct_linear_transform import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('data/K.txt')
        self.XY = np.loadtxt("data/platform_corners_metric.txt")
        # self.XY = np.loadtxt("data/platform_corners_metric_t23.txt") # Task 2.3
        
        self.platform_to_camera = np.loadtxt('data/platform_to_camera.txt')

        #self.lengths = np.array([11.45, 32.5, 5, 65, 3])*1e-2  # Task 1
        self.lengths = np.array([0.11249212, 0.32140264, 0.0507733,  0.66119392, 0.03120512]) #Task 3.1.1


        #self.heli_points = np.loadtxt('data/heli_points.txt').T # Task1
        self.heli_points = np.loadtxt('data/improved_marker_points.txt').T #Task 3.1.1



    def residuals(self, uv, weights, yaw, pitch, roll):
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
       
        r = np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]])

        return r[0]
        

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
        

            r = np.append(r, np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]]))

            
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




class ImprovedQuanser:
    def __init__(self):
        self.K = np.loadtxt('data/K.txt')
        self.platform_to_camera = np.loadtxt('data/platform_to_camera.txt')

        #self.heli_points = np.loadtxt('data/heli_points.txt').T
        #self.heli_points = np.loadtxt('data/improved_marker_points.txt').T
        self.heli_points = np.loadtxt('data/marker_points_best.txt').T # Task 3.1.2

        #self.lengths = np.array([0.11249211, 0.32140264, 0.05077297, 0.6611940541181118, 0.03120462792288483])
        #self.lengths = np.array([0.109862, 0.32139012, 0.05091132, 0.66160684, 0.03061243])
        #self.lengths = np.array([0.11037847,0.3218341, 0.05072337,  0.66186831,  0.03161999])
        self.lengths = np.array([0.11037847,  0.3218341 ,  0.05072337,  0.66186831,  0.03161999, -0.00233882,  0.01474224]) # Task 3.1.2
        self.stat_angles = np.array([-0.00251951,  0.00820128, -0.00647965,  0.03738877, -0.00275378, -0.03104896]) # Task 3.1.2


    def residuals(self, uv, weights, yaw, pitch, roll):
        # Compute the helicopter coordinate frames
        
        base_to_platform = rotate_x(self.stat_angles[0])@rotate_y(self.stat_angles[1]) @ translate(self.lengths[0]/2, self.lengths[0]/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = rotate_x(self.stat_angles[2])@rotate_z(self.stat_angles[3]) @ translate(self.lengths[5], 0.00,  self.lengths[1])@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -self.lengths[2])
        rotors_to_arm    = rotate_y(self.stat_angles[4])@rotate_z(self.stat_angles[5]) @ translate(self.lengths[3], self.lengths[6], -self.lengths[4])@rotate_x(roll)
        
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()
        
        r = np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]])

        return r[0]



    def residuals3(self, detections, p):
        # Compute the helicopter coordinate frames

        lengths = p[:7]
        markers = p[7:28]
        stat_angles = p[28:34]
        trajectory = p[34:]
        markers = np.vstack((markers.reshape(7,3).T,np.ones(7)))
        
        num_images = detections.shape[0]

        r = np.array([])

        for image_number in range(num_images):

            uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
            weights = detections[image_number, ::3]

            base_to_platform = rotate_x(stat_angles[0])@rotate_y(stat_angles[1]) @ translate(lengths[0]/2, lengths[0]/2, 0.0)@rotate_z(trajectory[image_number*3])
            hinge_to_base    = rotate_x(stat_angles[2])@rotate_z(stat_angles[3]) @ translate(lengths[5], 0.00,  lengths[1])@rotate_y(trajectory[image_number*3+1])
            arm_to_hinge     = translate(0.00, 0.00, -lengths[2])
            rotors_to_arm    = rotate_y(stat_angles[4])@rotate_z(stat_angles[5]) @ translate(lengths[3], lengths[6], -lengths[4])@rotate_x(trajectory[image_number*3+2])
            base_to_camera   = self.platform_to_camera@base_to_platform
            hinge_to_camera  = base_to_camera@hinge_to_base
            arm_to_camera    = hinge_to_camera@arm_to_hinge
            rotors_to_camera = arm_to_camera@rotors_to_arm

            p1 = arm_to_camera @ markers[:,:3]
            p2 = rotors_to_camera @ markers[:,3:]
            uv_hat = project(self.K, np.hstack([p1, p2]))

            r = np.append(r, np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]]))

        return r


    def residuals_individual(self, detections, p, image_number): # completing the right block. 3l x 2nl
        # Compute the helicopter coordinate frames

        lengths = p[:7]
        markers = p[7:28]
        stat_angles = p[28:34]
        trajectory = p[34:]
        markers = np.vstack((markers.reshape(7,3).T,np.ones(7)))

        r = np.array([])

        uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
        weights = detections[image_number, ::3]

        base_to_platform = rotate_x(stat_angles[0])@rotate_y(stat_angles[1]) @ translate(lengths[0]/2, lengths[0]/2, 0.0)@rotate_z(trajectory[0])
        hinge_to_base    = rotate_x(stat_angles[2])@rotate_z(stat_angles[3]) @ translate(lengths[5], 0.00,  lengths[1])@rotate_y(trajectory[1])
        arm_to_hinge     = translate(0.00, 0.00, -lengths[2])
        rotors_to_arm    = rotate_y(stat_angles[4])@rotate_z(stat_angles[5]) @ translate(lengths[3], lengths[6], -lengths[4])@rotate_x(trajectory[2])
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        p1 = self.arm_to_camera @ markers[:,:3]
        p2 = self.rotors_to_camera @ markers[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()

        r = np.append(r, np.hstack([(uv_hat[0]-uv[0])*weights[None,:], (uv_hat[1]-uv[1])*weights[None,:]]))
       
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

