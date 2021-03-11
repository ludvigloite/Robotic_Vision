import matplotlib.pyplot as plt
import numpy as np
from direct_linear_transform import *

K           = np.loadtxt('../data/K.txt')
detections  = np.loadtxt('../data/detections.txt')
XY          = np.loadtxt('../data/XY.txt').T
n_total     = XY.shape[1] # Total number of markers (= 24)

fig = plt.figure(figsize=plt.figaspect(0.35))

# for image_number in range(23): # Use this to run on all images
for image_number in [4, 5 ,21]: # Use this to run on a single image

    # Load data
    # valid : Boolean mask where valid[i] is True if marker i was detected
    #     n : Number of successfully detected markers (<= n_total)
    #    uv : Pixel coordinates of successfully detected markers
    valid = detections[image_number, 0::3] == True
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    uv = uv[:, valid]
    n = uv.shape[1]

    # Tip: The 'valid' array can be used to perform Boolean array indexing,
    # e.g. to extract the XY values of only those markers that were detected.
    # Use this when calling estimate_H and when computing reprojection error.

    # Tip: Helper arrays with 0 and/or 1 appended can be useful if
    # you want to replace for-loops with array/matrix operations.
    # uv1 = np.vstack((uv, np.ones(n)))
    # XY1 = np.vstack((XY, np.ones(n_total)))
    # XY01 = np.vstack((XY, np.zeros(n_total), np.ones(n_total)))

    u_tilde = np.append(uv, np.ones(n)).reshape((3,n))  # TASK: Compute calibrated image coordinates
    xy_tilde = np.linalg.inv(K) @ u_tilde
    xy = np.array([xy_tilde[0]/xy_tilde[2], xy_tilde[1]/xy_tilde[2]])
    H = estimate_H(xy, XY[:, valid])   # TASK: Implement this function
    
    projected_xy = H @ np.append(XY[:, valid], np.ones(n)).reshape((3,n))
    uv_from_H_hom = K @ projected_xy # TASK: Compute predicted pixel coordinates using H
    uv_from_H = np.array([uv_from_H_hom[0]/uv_from_H_hom[2], uv_from_H_hom[1]/uv_from_H_hom[2]])

    [avg_rep_err, max_rep_err, min_rep_err] = calc_reprojection_error(uv, uv_from_H)
    print("Image nr {}".format(image_number))
    print("Average reprojection error:\t{}\nMaximum reprojection error:\t{}\nMinimum reprojection error:\t{}".format(avg_rep_err, max_rep_err, min_rep_err))

    T, Q = decompose_H(H) # TASK: Implement this function

    det_Q = np.abs(np.linalg.det(Q))
    det_T = np.abs(np.linalg.det(T[:3,:3]))
    print("Pure rotational properties:\t{}".format(det_Q))
    print("Refined rotational properties:\t{}".format(det_T))
    print("")

    # The figure should be saved in the data directory as out0000.png, etc.
    # NB! generate_figure expects the predicted pixel coordinates as 'uv_from_H'.
    plt.clf()
    generate_figure(fig, image_number, K, T, uv, uv_from_H, XY)
    plt.savefig('../data/out%04d.png' % image_number)
