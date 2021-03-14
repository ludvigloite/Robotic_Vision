import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser
from generate_quanser_summary import *

detections = np.loadtxt('data/detections.txt')
heli_points = np.loadtxt('data/heli_points.txt')


# The script runs up to, but not including, this image.
# run_until = 87 # Task 1.3
# run_until = 88 # Task 1.4
# run_until = 1 # Task 1.5
run_until = detections.shape[0] # Task 1.7

# Change this if you want the Quanser visualization for a different image.
# (Can be useful for Task 1.4)
visualize_number = 0

quanser = Quanser()

# Initialize the parameter vector
p = np.array([11.6, 28.9, 0.0])*np.pi/180 # Optimal for image number 0
#p = np.array([0.0, 0.0, 0.0]) # For Task 1.5

all_residuals = []
init_trajectory = np.zeros(run_until * 3)
printbool = False

for image_number in range(run_until):
    weights = detections[image_number, ::3]
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    residualsfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

    p = levenberg_marquardt(residualsfun,p)
    init_trajectory[image_number * 3:image_number*3+3] = p
    #init_trajectory[image_number, :] = p
    #print(p)
    #print(trajectory[image_number * 3:image_number*3+3])

print("Finished computing init trajectory")

init_lengths = [0.1145, 0.325, 0.05, 0.65, 0.03]
init_markers = heli_points[:,:3].flatten()


p = np.hstack([init_lengths,init_markers,init_trajectory])
weights = detections[:, ::3]
uv = np.vstack((detections[:, 1::3], detections[:, 2::3]))


residualsfun2 = lambda p : quanser.residuals3(detections, p)
residualsfun_individual = lambda p, image_number : quanser.residuals_individual(detections, p,image_number)


m = len(init_lengths) + 7*3 # num static variables (lengths + markers in 3D)

p = levenberg_marquardt2(residualsfun2, residualsfun_individual, p, m)

print(f"\nLengths:\n {p[:5]}")
print(f"\nMarkers:\n {p[5:26].reshape(7,3)}")

