import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser, ImprovedQuanser
from generate_quanser_summary import *

detections = np.loadtxt('data/detections.txt')

# The script runs up to, but not including, this image.
# run_until = 87 # Task 1.3
# run_until = 88 # Task 1.4
# run_until = 1 # Task 1.5
run_until = detections.shape[0] # Task 1.7

# Change this if you want the Quanser visualization for a different image.
# (Can be useful for Task 1.4)
visualize_number = 0

quanser = Quanser()
improvedQuanser = ImprovedQuanser()

# Initialize the parameter vector
p = np.array([11.6, 28.9, 0.0])*np.pi/180 # Optimal for image number 0
#p = np.array([0.0, 0.0, 0.0]) # For Task 1.5

all_residuals = []
trajectory = np.zeros((run_until, 3))
printbool = False

for image_number in range(run_until):
    weights = detections[image_number, ::3]
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    print("im nr: ", image_number)

    #Task 1 and 3.1.1(dependant on lengths and markers)
    #residualsfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

    #Task 3.1.2
    residualsfun = lambda p : improvedQuanser.residuals(uv, weights, p[0], p[1], p[2])

    # Task 1.3:
    # Implement gauss_newton (see methods.py).
    
    """ Task 1.4
    if image_number > 80:
        printbool = True
        print("image nr: ", image_number)
    """

    #p = gauss_newton(residualsfun, p, printbool)

    p = levenberg_marquardt(residualsfun,p)

    r = residualsfun(p)
    all_residuals.append(r)
    trajectory[image_number, :] = p
    if image_number == visualize_number:
        print('Residuals on image number', image_number, r)
        #quanser.draw(uv, weights, image_number) #Task 1 or 3.1.1
        improvedQuanser.draw(uv, weights, image_number) # Task 3.1.2
        

generate_quanser_summary(trajectory, all_residuals, detections)
plt.show()
