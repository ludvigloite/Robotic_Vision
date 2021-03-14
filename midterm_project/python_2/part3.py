import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser
from generate_quanser_summary import *
from common import *
from extract_trajectory import get_trajectory

lengths = np.array([11.45, 32.5, 5, 65, 3])*1e-2 # Initial length values
heli_points = np.loadtxt('midterm_project/data/heli_points.txt') # Initial marker locations
trajectory = get_trajectory() # Initial trajectory (angle) values
print(heli_points.T)
detections = np.loadtxt('midterm_project/data/detections.txt')
K = np.loadtxt('midterm_project/data/K.txt')
platform_to_camera = np.loadtxt('midterm_project/data/platform_to_camera.txt')

def calculate_uv_hat(p, angles, K):
    base_to_platform = translate(p[0]/2,p[0]/2, 0.0) @ rotate_z(angles[0])
    hinge_to_base    = translate(0.00, 0.00,  p[1]) @ rotate_y(angles[1])
    arm_to_hinge     = translate(0.00, 0.00, -p[2])
    rotors_to_arm    = translate(p[3], 0.00, -p[4]) @ rotate_x(angles[2])
    base_to_camera   = platform_to_camera @ base_to_platform
    hinge_to_camera  = base_to_camera @ hinge_to_base
    arm_to_camera    = hinge_to_camera @ arm_to_hinge
    rotors_to_camera = arm_to_camera @ rotors_to_arm

    # Compute the predicted image location of the markers
    markers_arm = np.vstack((np.reshape(p[5:14], (3,3)).T, np.ones(3)))
    markers_rotor = np.vstack((np.reshape(p[14:26], (4,3)).T, np.ones(4)))
    p1 = arm_to_camera @ markers_arm
    p2 = rotors_to_camera @ markers_rotor
    uv_hat = project(K, np.hstack([p1, p2]))

    return uv_hat
    

def residuals3(uv, weights, p, K, n, m, l):
    angles = p[m:]
    angles = angles.reshape((angles.shape[0]//3,3))
    rs = np.array([])
    
    for j in range(angles.shape[0]):
        uv_hat = calculate_uv_hat(p[:m], angles[j], K)
        r = np.zeros(2*n)

        for i in range(n):
            r[i] = (uv_hat[0][i] - uv[j][i]) * weights[j][i]
            r[n+i] = (uv_hat[1][i] - uv[l+j][i]) * weights[j][i]
        rs = np.append(rs, r)
        
    
    return rs

def residuals4(uv, weights, p, K, image_number, n, m, l):
    angles = p[m:]
    angles = angles.reshape((angles.shape[0]//3,3))
    rs = np.array([])
    
    for j in range(1):
        uv_hat = calculate_uv_hat(p[:m], angles[j], K)
        r = np.zeros(2*n)

        for i in range(n):
            r[i] = (uv_hat[0][i] - uv[image_number][i]) * weights[image_number][i]
            r[n+i] = (uv_hat[1][i] - uv[l+image_number][i]) * weights[image_number][i]
        rs = np.append(rs, r)
    
    return rs


heli_points_flattened = np.ndarray.flatten(heli_points[:,:3])
trajectory_flattened = np.ndarray.flatten(trajectory)
p = np.concatenate((lengths, heli_points_flattened, trajectory_flattened))

weights = detections[:, ::3]
uv = np.vstack((detections[:, 1::3], detections[:, 2::3]))

residualsfun = lambda p, n, m, l : residuals3(uv, weights, p, K, n, m, l)
residualsfun2 = lambda p, image_number, n, m, l : residuals4(uv, weights, p, K, image_number, n, m, l)

n = heli_points.shape[0] # num markers
m = len(lengths) + 7*3 # num static variables (lengths + markers in 3D)
l = trajectory.shape[0] # num dynamic variables

p = LM_improved(residualsfun, residualsfun2, p, n, m, l)


print(lengths)
print(np.round(p[:5],3))
print("___________")
print(np.round(p[5:26].reshape(7,3).T,3))
print(heli_points[:,:3].T)
