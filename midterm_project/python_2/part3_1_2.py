import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser2 import Quanser2
from generate_quanser_summary import *
from common import *
from extract_trajectory import get_trajectory

lengths = np.array([11.45, 32.5, 5, 65, 3, 0, 0])*1e-2 # Initial length values
static_angles = np.zeros(6) # Initial static angles
heli_points = np.loadtxt('midterm_project/data/heli_points.txt') # Initial marker locations
trajectory = get_trajectory() # Initial trajectory (angle) values

detections = np.loadtxt('midterm_project/data/detections.txt')
K = np.loadtxt('midterm_project/data/K.txt')
platform_to_camera = np.loadtxt('midterm_project/data/platform_to_camera.txt')

quanser2 = Quanser2()

heli_points_flattened = np.ndarray.flatten(heli_points[:,:3])
trajectory_flattened = np.ndarray.flatten(trajectory)
p = np.concatenate((lengths, heli_points_flattened, static_angles, trajectory_flattened))

weights = detections[:, ::3]
uv = np.vstack((detections[:, 1::3], detections[:, 2::3]))

residualsfun = lambda p, n, m, l : quanser2.residuals3(uv, weights, p, K, n, m, l)
residualsfun2 = lambda p, image_number, n, m, l : quanser2.residuals4(uv, weights, p, K, image_number, n, m, l)

n = heli_points.shape[0] # num markers
m = len(lengths) + n*3 + static_angles.shape[0] # num static variables (lengths + markers in 3D)
l = trajectory.shape[0] # num dynamic variables

p = LM_improved(residualsfun, residualsfun2, p, n, m, l)

print(lengths)
print(np.round(p[:7],3))
print("___________")
print(np.round(p[7:28].reshape(7,3).T,3))
print(heli_points[:,:3].T)
