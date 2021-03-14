import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import ImprovedQuanser, Quanser
from generate_quanser_summary import *

detections = np.loadtxt('data/detections.txt')
heli_points = np.loadtxt('data/heli_points.txt')

run_until = detections.shape[0]

visualize_number = 0

improvedQuanser = ImprovedQuanser()
quanser = Quanser()

p = np.array([11.6, 28.9, 0.0])*np.pi/180 # Optimal for image number 0

init_trajectory = np.zeros(run_until * 3)
printbool = False

for image_number in range(run_until):
    weights = detections[image_number, ::3]
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    residualsfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

    p = levenberg_marquardt(residualsfun,p)
    init_trajectory[image_number * 3:image_number*3+3] = p
    

print("Finished computing init trajectory")

init_lengths = [0.1145, 0.325, 0.05, 0.65, 0.03, 0, 0]
init_stat_angles = np.zeros(6)
init_markers = heli_points[:,:3].flatten()


p = np.hstack([init_lengths, init_markers, init_stat_angles, init_trajectory])
weights = detections[:, ::3]
uv = np.vstack((detections[:, 1::3], detections[:, 2::3]))


residualsfun2 = lambda p : improvedQuanser.residuals3(detections, p)
residualsfun_individual = lambda p, image_number : improvedQuanser.residuals_individual(detections, p,image_number)

m = len(init_lengths) + 7*3 + init_stat_angles.shape[0]

p = levenberg_marquardt2(residualsfun2, residualsfun_individual, p, m)

print(f"\nLengths:\n {p[:7]}")
print(f"\nMarkers:\n {p[7:28].reshape(7,3)}")
print(f"\nStatic Angles:\n {p[28:34]}")

"""

Iterations: 60

Lengths:
[ 0.109862    0.32139012  0.05091132  0.66160684  0.03061243 -0.00259357 0.0037901 ]

Markers:
 [[-0.13053376 -0.02438211  0.00670227]
 [ 0.17817365 -0.02209708  0.01064194]
 [ 0.4339905  -0.02010201  0.01266714]
 [-0.03106559 -0.08698486 -0.03808854]
 [-0.02950414 -0.17350838 -0.06374418]
 [-0.03430902  0.21272357 -0.02141904]
 [-0.03351442  0.10932655 -0.02474112]]

Static Angles:
 [-1.88732260e-03  8.69393423e-03 -5.11582505e-02  2.13863939e-02
 -8.96868891e-05 -1.53384656e-02]

"""
