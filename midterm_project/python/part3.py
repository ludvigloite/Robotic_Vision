import matplotlib.pyplot as plt
import numpy as np
from methods_part3 import *
from quanser_part3 import Quanser
from generate_quanser_summary import *

if __name__ == "__main__":

    detections = np.loadtxt('data/detections.txt')

    # The script runs up to, but not including, this image.
    run_until = detections.shape[0]

    # Change this if you want the Quanser visualization for a different image.
    visualize_number = 0

    quanser = Quanser()

    # Initialize the parameter vector
    lengths = np.array([0.1145, 0.325, 0.050, 0.65, 0.030])
    markers = np.loadtxt('data/heli_points.txt').T
    angles = np.ones([detections.shape[0], 3])*np.array([11.6, 28.9, 0.0])*np.pi/180 # Optimal for image number 0
    # angles = np.array([0.0, 0.0, 0.0]) # For Task 1.5
    p = np.array([lengths, markers, angles])


    all_residuals = []
    trajectory = np.zeros((run_until, 3))
    printbool = False

    # Load all data
    for image_number in range(run_until): # Don't need
        weights = detections[:, ::3]
        uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))

        # Tip:
        # 'uv' is a 2x7 array of detected marker locations.
        # It is the same size in each image, but some of its
        # entries may be invalid if the corresponding markers were
        # not detected. Which entries are valid is encoded in
        # the 'weights' array, which is a 1D array of length 7.

        # Tip:
        # Make your optimization method accept a lambda function
        # to compute the vector of residuals. You can then reuse
        # the method later by passing a different lambda function.
        residualsfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

        p = levenberg_marquardt(residualsfun,p)

        # Note:
        # The plotting code assumes that p is a 1D array of length 3
        # and r is a 1D array of length 2n (n=7), where the first
        # n elements are the horizontal residual components, and
        # the last n elements the vertical components.

        r = residualsfun(p)
        all_residuals.append(r)
        trajectory[image_number, :] = p
        if image_number == visualize_number:
            print('Residuals on image number', image_number, r)
            quanser.draw(uv, weights, image_number)

    # Note:
    # The generated figures will be saved in your working
    # directory under the filenames out_*.png.

    generate_quanser_summary(trajectory, all_residuals, detections)
    plt.show()
