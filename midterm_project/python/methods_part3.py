import numpy as np


def jacobian(residualsfun, p, epsilon):
    n = residualsfun(p).shape[0]
    m = p.shape[0]
    J = np.zeros([n,m]) # task 2.2
    for i in range(m):
        e = np.zeros(m)
        e[i] = epsilon
        J[:,i] = (residualsfun(p+e)-residualsfun(p-e)) / (2*epsilon)

    return J



def levenberg_marquardt(residualsfun, p0, printbool=False, stop_precision = 1e-6, num_iterations=100, finite_difference_epsilon=1e-5):
    p = p0
    e = finite_difference_epsilon
    J = jacobian(residualsfun,p,e)
    dim = J.shape[1]
    mu = np.max(np.diag((J.T @ J))) * 1e-3

    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        J = jacobian(residualsfun,p,e)

        # 2: Form the normal equation terms JTJ and JTr.
        if printbool:
            print("J= \n",J)
            print(f"JTJ = \n{J.T @ J}")
        
        JTJ = J.T @ J
        JTr = J.T @ residualsfun(p)

        delta = -np.linalg.inv(JTJ + mu * np.eye(dim)) @ JTr

        if printbool:
            print(f"delta  = {delta}")

        E_delta = np.sum(residualsfun(p+delta)**2)
        E = np.sum(residualsfun(p)**2)

        if E_delta < E:
            mu /= 3
            p += delta
        else:
            mu *= 2

        if np.linalg.norm(delta) < stop_precision:
            return p

    return p
