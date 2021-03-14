import numpy as np
from scipy.linalg import block_diag

# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.
def gauss_newton(residualsfun, p0, step_size=0.25, num_iterations=100, finite_difference_epsilon=1e-5):
    # See the comment in part1.py regarding the 'residualsfun' argument.

    p = p0.copy()
    eps = finite_difference_epsilon
    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        
        """        print('now')
        r = residualsfun(p)

        J = np.zeros((r.shape[0], p.shape[0]))
        J[:,0] = (residualsfun(p+np.array([eps, 0, 0])) - residualsfun(p-np.array([eps, 0, 0]))) / (2*eps)
        J[:,1] = (residualsfun(p+np.array([0, eps, 0])) - residualsfun(p-np.array([0, eps, 0]))) / (2*eps)
        J[:,2] = (residualsfun(p+np.array([0, 0, eps])) - residualsfun(p-np.array([0, 0, eps]))) / (2*eps)

        # 2: Form the normal equation terms JTJ and JTr.
        JTJ = J.T @ J
        JTr = J.T @ r
   
        # 3: Solve for the step delta and update p as
        delta = np.linalg.solve(JTJ, -JTr)
        p = p + delta"""
        pass
    return p

# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.
def levenberg_marquardt(residualsfun, p0, num_iterations=1000, finite_difference_epsilon=1e-5):
    p = p0.copy()
    #iteration = 0
    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        r = residualsfun(p)
        
        J = np.zeros((r.shape[0], p.shape[0]))
        for i in range(p.shape[0]):
            eps = np.zeros(p.shape[0])
            eps[i] = finite_difference_epsilon
            J[:,i] = (residualsfun(p+eps) - residualsfun(p-eps)) / (2*finite_difference_epsilon)

        # 2: Form the normal equation terms JTJ and JTr.
        JTJ = J.T @ J
        JTr = J.T @ r
        if iteration == 0:
            mu = max(np.diagonal(JTJ))*1e-3
   
        # 3: Solve for the step delta and update p as
        #print(sum(r**2))
        delta = np.linalg.solve(JTJ + mu*np.eye(JTJ.shape[0]), -JTr)
        if sum(residualsfun(p + delta)**2) < sum(r**2):
            p = p + delta
            mu = mu/3
        else:
            mu = mu*2
        
        if np.linalg.norm(delta) < 1e-3:
            # print("Stop criteria satisfied")
            # print("Iteration: ", iteration)
            return p
    print("Stop criteria not satisfied")
    print("Iteration: ", iteration)   
    return p


def LM_improved(residualsfun, residualsfun2, p0, n, m, l, num_iterations=200, finite_difference_epsilon=1e-5):
    p = p0.copy()

    for iteration in range(num_iterations):
        r = residualsfun(p, n, m, l)

        J1 = np.zeros((r.shape[0], m))
        for i in range(m):
            eps = np.zeros(p.shape[0])
            eps[i] = finite_difference_epsilon
            J1[:,i] = (residualsfun(p+eps, n, m, l) - residualsfun(p-eps, n, m, l)) / (2*finite_difference_epsilon)
        

        J2 =  np.zeros((l, 2*n, 3))
        for i in range(l):
            Jx = np.zeros((2*n, 3))
            for j in range(3):
                px = np.concatenate((p[:m], p[(m + 3*i):(m + 3*(i+1))]))
                eps = np.zeros(m+3)
                eps[m+j] = finite_difference_epsilon
                Jx[:,j] = (residualsfun2(px+eps, i, n, m, l) - residualsfun2(px-eps, i, n, m, l)) / (2*finite_difference_epsilon)
            J2[i] = Jx
        
        J2_dyn = block_diag(*J2)
        
        A11 = J1.T @ J1
        A12 = J1.T @ J2_dyn
        A21 = A12.T
        A22 = J2_dyn.T @ J2_dyn

        if iteration == 0:
            mu = max(np.amax(np.diagonal(A11)), np.amax(np.diagonal(A22)))*1e-3

        A11 += mu*np.eye(A11.shape[0])
        A22 += mu*np.eye(A22.shape[0])

        D_inv = np.linalg.inv(A22)

        a = -(J1.T @ r)
        b = -(J2_dyn.T @ r)
        
        delta_static = np.linalg.solve(A11 - A12 @ D_inv @ A21, a - A12 @ D_inv @ b)
        delta_dyn = np.linalg.solve(A22, b - A21 @ delta_static)

        delta = np.hstack((delta_static, delta_dyn))

        print("Iteration: ", iteration, "E(p): ", np.round(sum(residualsfun(p + delta, n, m, l)**2),3), "|delta|: ", np.round(np.linalg.norm(delta),3))
        if sum(residualsfun(p + delta, n, m, l)**2) < sum(r**2):
            p = p + delta
            mu = mu/3
            print("Step accepted")
            
        else:
            mu = mu*2
            print("Step rejected")
        print("___________")
        if np.linalg.norm(delta) < 1e-3:
            print("Stop criteria satisfied")
            print("Iteration: ", iteration)
            return p
    print("Stop criteria not satisfied")
    print("Iteration: ", iteration)   
    return p