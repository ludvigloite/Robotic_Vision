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
Lengths:
[0.11037847,  0.3218341 ,  0.05072337,  0.66186831,  0.03161999, -0.00233882,  0.01474224]

Markers:
    [[-0.13064549, -0.00979059,  0.00658394],
       [ 0.17822714, -0.01220431,  0.0110655 ],
       [ 0.43418592, -0.01417044,  0.01353566],
       [-0.03100851, -0.08856472, -0.03438397],
       [-0.02945179, -0.17614164, -0.05638121],
       [-0.03422383,  0.21175494, -0.03036017],
       [-0.03344751,  0.10824166, -0.02932079]]

Static Angles:
[-0.00251951  0.00820128 -0.00647965  0.03738877 -0.00275378 -0.03104896]



"""
