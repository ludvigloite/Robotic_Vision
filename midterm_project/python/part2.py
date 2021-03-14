import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser
from direct_linear_transform import *
from generate_quanser_summary import *

if __name__ == "__main__":

    XY = np.loadtxt("data/platform_corners_metric.txt")
    uv = np.loadtxt("data/platform_corners_image.txt")
    # XY = np.loadtxt("data/platform_corners_metric_t23.txt") # Task 2.3
    # uv = np.loadtxt("data/platform_corners_image_t23.txt") # Task 2.3

    K = np.loadtxt("data/K.txt")
    T_truth = np.loadtxt("data/platform_to_camera.txt")

    u_tilde = np.append(uv, np.ones(uv.shape[1])).reshape((3,uv.shape[1]))
    xy_tilde = np.linalg.inv(K) @ u_tilde
    xy = np.array([xy_tilde[0]/xy_tilde[2], xy_tilde[1]/xy_tilde[2]])
    H = estimate_H(xy, XY)

    # Part 2.1.a
    uv_tilde_a = K @ H @ XY[np.array([True,True,False,True])]
    uv_a = np.array([uv_tilde_a[0]/uv_tilde_a[2], uv_tilde_a[1]/uv_tilde_a[2]])

    # Part 2.1.b
    T = decompose_H(H)[np.array([True, True, True, False])]
    uv_tilde_b = K @ T @ XY
    uv_b = np.array([uv_tilde_b[0]/uv_tilde_b[2], uv_tilde_b[1]/uv_tilde_b[2]])

    fig = plt.figure(figsize=plt.figaspect(0.35))
    T = decompose_H(H)
    print("Task 2.1.a")
    [avg_rep_err_a, max_rep_err_a, min_rep_err_a] = calc_reprojection_error(uv, uv_a)
    print("Average reprojection error:\t{}\nMaximum reprojection error:\t{}\nMinimum reprojection error:\t{}".format(avg_rep_err_a, max_rep_err_a, min_rep_err_a))
    print("Point reprojection error:\t{}".format(np.linalg.norm(uv-uv_a, axis=0)))
    generate_figure(fig, 0, K, T, uv, uv_a, XY)
    # plt.show()
    plt.savefig("task21a")

    plt.clf()
    print("Task 2.1.b")
    [avg_rep_err_b, max_rep_err_b, min_rep_err_b] = calc_reprojection_error(uv, uv_b)
    print("Average reprojection error:\t{}\nMaximum reprojection error:\t{}\nMinimum reprojection error:\t{}".format(avg_rep_err_b, max_rep_err_b, min_rep_err_b))
    print("Point reprojection error:\t{}".format(np.linalg.norm(uv-uv_b, axis=0)))
    generate_figure(fig, 1, K, T, uv, uv_b, XY)
    # plt.show()
    plt.savefig("task21b")


    # Part 2.2 - Levenberg Marquardt
    quanser = Quanser()

    # Initialize the parameter vector
    weights = np.ones(4)
    t1, t2, t3 = T[:3,3]
    p = np.array([0.0, 0.0, 0.0, t1, t2, t3]) # For Task 1.5

    run_until = 1
    all_residuals = []
    trajectory = np.zeros((run_until, 3))
    printbool = False

    R0 = T[:3,:3]
    residualsfun = lambda p : quanser.residuals2(uv, R0, p)

    p = levenberg_marquardt(residualsfun, p)
    r = residualsfun(p)
    optimized_T = quanser.get_optimized_T()
    all_residuals.append(r)
    print("\nTask 2.2")
    print("DLT T:\n{}\nOptimized T:\n{}\nTruth:\n{}\n".format(T, optimized_T, T_truth))
    print("Reprojection error Levenberg Marquardt:\t{}".format(r))

    
    uv_tilde_opt = K @ optimized_T[np.array([True, True, True, False])] @ XY
    uv_opt = np.array([uv_tilde_opt[0]/uv_tilde_opt[2], uv_tilde_opt[1]/uv_tilde_opt[2]])

    plt.clf()
    generate_figure(fig, 1, K, optimized_T, uv, uv_opt, XY)
    # plt.show()
    plt.savefig("task22")
    