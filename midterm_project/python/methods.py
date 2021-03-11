import numpy as np

# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.
def gauss_newton(residualsfun, p0, step_size=0.25, num_iterations=100, finite_difference_epsilon=1e-5):
    # See the comment in part1.py regarding the 'residualsfun' argument.

    p = p0.copy()
    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        epsilon = np.array([1, 0, 0]) * finite_difference_epsilon
        J1 = (residualsfun(p+epsilon) - residualsfun(p-epsilon)) / (2*finite_difference_epsilon)
        epsilon = np.array([0, 1, 0]) * finite_difference_epsilon
        J2 = (residualsfun(p+epsilon) - residualsfun(p-epsilon)) / (2*finite_difference_epsilon)
        epsilon = np.array([0, 0, 1]) * finite_difference_epsilon
        J3 = (residualsfun(p+epsilon) - residualsfun(p-epsilon)) / (2*finite_difference_epsilon)

        J = np.vstack([J1, J2, J3]).T

        # 2: Form the normal equation terms JTJ and JTr.
        delta = - np.linalg.inv(J.T @ J) @ J.T @ residualsfun(p)

        # 3: Solve for the step delta and update p as
        p = p + step_size*delta
    return p

# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.
def levenberg_marquardt(residualsfun, p0, termination_cond=1e-3, num_iterations=100, finite_difference_epsilon=1e-5):
    # Initial values
    p = p0.copy()
    e1 = np.array([1, 0, 0]) * finite_difference_epsilon
    e2 = np.array([0, 1, 0]) * finite_difference_epsilon
    e3 = np.array([0, 0, 1]) * finite_difference_epsilon
    J1 = (residualsfun(p+e1) - residualsfun(p-e1)) / (2*finite_difference_epsilon)
    J2 = (residualsfun(p+e2) - residualsfun(p-e2)) / (2*finite_difference_epsilon)
    J3 = (residualsfun(p+e3) - residualsfun(p-e3)) / (2*finite_difference_epsilon)

    J = np.vstack([J1, J2, J3]).T
    mu = np.max((J.T@J).diagonal()) * 1e-3

    for iteration in range(num_iterations):
        e1 = np.array([1, 0, 0]) * finite_difference_epsilon
        e2 = np.array([0, 1, 0]) * finite_difference_epsilon
        e3 = np.array([0, 0, 1]) * finite_difference_epsilon
        J1 = (residualsfun(p+e1) - residualsfun(p-e1)) / (2*finite_difference_epsilon)
        J2 = (residualsfun(p+e2) - residualsfun(p-e2)) / (2*finite_difference_epsilon)
        J3 = (residualsfun(p+e3) - residualsfun(p-e3)) / (2*finite_difference_epsilon)

        J = np.vstack([J1, J2, J3]).T

        delta = - np.linalg.inv(J.T @ J + mu * np.eye(J.shape[1])) @ J.T @ residualsfun(p)

        err_p = (residualsfun(p)**2).sum()
        err_d = ((residualsfun(p) + J @ delta)**2).sum()

        # print("Iteration: {}\nmu: {}\npsi: {}\ntheta: {}\nphi: {}\n".format(iteration, mu, p[0], p[1], p[2]))

        if (np.linalg.norm(delta) < termination_cond):
            p = p + delta
            break
        elif (err_d < err_p):
            mu /= 3
            p += delta
        else:
            mu *= 2
            

    return p
