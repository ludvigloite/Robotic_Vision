import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser
from generate_quanser_summary import *



def get_trajectory():
    detections = np.loadtxt('../data/detections.txt')
    run_until = detections.shape[0]
    quanser = Quanser()
    p = np.array([11.6, 28.9, 0.0])*np.pi/180 # Optimal for image number 0

    trajectory = np.zeros((run_until, 3))
    for image_number in range(run_until):
        weights = detections[image_number, ::3]
        uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))

        residualsfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])
        p = levenberg_marquardt(residualsfun, p)

        trajectory[image_number, :] = p
    
    print("Trajectory extracted")
    print("_____________________")
    
    return trajectory
