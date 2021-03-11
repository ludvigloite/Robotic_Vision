import numpy as np


def jacobian(residualsfun, p, epsilon):
    J = np.zeros([14,3])
    e1 = np.array([1,0,0])*epsilon
    e2 = np.array([0,1,0])*epsilon
    e3 = np.array([0,0,1])*epsilon

    J[:,0] = (residualsfun(p+e1)-residualsfun(p-e1)) / (2*epsilon)
    J[:,1] = (residualsfun(p+e2)-residualsfun(p-e2)) / (2*epsilon)
    J[:,2] = (residualsfun(p+e3)-residualsfun(p-e3)) / (2*epsilon)

    return J


# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.
def gauss_newton(residualsfun, p0, printbool, step_size=0.25, num_iterations=100, finite_difference_epsilon=1e-5):
    # See the comment in part1.py regarding the 'residualsfun' argument.
    #print("res: ", residualsfun(p0))
    #print("p0: ",p0)
    e = finite_difference_epsilon
    p = p0.copy()
    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.

        J = np.zeros([14,3])
        e1 = np.array([1,0,0])*e
        e2 = np.array([0,1,0])*e
        e3 = np.array([0,0,1])*e

        J[:,0] = (residualsfun(p+e1)-residualsfun(p-e1)) / (2*e)
        J[:,1] = (residualsfun(p+e2)-residualsfun(p-e2)) / (2*e)
        J[:,2] = (residualsfun(p+e3)-residualsfun(p-e3)) / (2*e)

        # 2: Form the normal equation terms JTJ and JTr.

        if printbool:
            print("J= \n",J)
            print(f"JTJ = \n{J.T @ J}")
        

        JTJ = J.T @ J
        JTr = J.T @ residualsfun(p)

        delta = -np.linalg.inv(JTJ) @ JTr

        if printbool:
            print(f"delta  = {delta}")

        # 3: Solve for the step delta and update p as
        p += step_size*delta

        #pass # Remove me!

    return p


# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.
def levenberg_marquardt(residualsfun, p0, printbool=False, stop_precision = 1e-6, num_iterations=100, finite_difference_epsilon=1e-5):

    #p = [0,0,0] #task 1.5

    p = p0
    e = finite_difference_epsilon

    J = jacobian(residualsfun,p,e)

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

        delta = -np.linalg.inv(JTJ + mu * np.eye(3)) @ JTr

        if printbool:
            print(f"delta  = {delta}")

        
        E_delta = np.sum(residualsfun(p+delta)**2)
        E = np.sum(residualsfun(p)**2)

        if E_delta < E:
            mu /= 3
            p += delta
        else:
            mu *= 2

        #print(f"iter: {iteration} \n p: {p} \n mu: {mu}")

        if np.linalg.norm(delta) < stop_precision:
            return p

    return p
