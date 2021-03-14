import numpy as np
from scipy.linalg import block_diag


def jacobian(residualsfun, p, epsilon):
    # J = np.zeros([14,3])
    n = residualsfun(p).shape[0]
    m = p.shape[0]
    J = np.zeros((n,m)) # task 2.2
    for i in range(m):
        e = np.zeros(m)
        e[i] = epsilon
        J[:,i] = (residualsfun(p+e)-residualsfun(p-e)) / (2*epsilon)

    return J

def jacobian2(residualsfun, p, epsilon, l, m, n):
    
    J = np.zeros((residualsfun(p).shape[0],m))
    for i in range(m):
        e = np.zeros(p.shape[0])
        e[i] = epsilon
        J[:,i] = (residualsfun(p+e)-residualsfun(p-e)) / (2*epsilon)

    return J

def jacobian3(residualsfun, p, epsilon, l, m, n):
    
    J = np.zeros((l,2*n,3))
    for image in range(l):
        Ji = np.zeros((2*n, 3))
        for j in range(3):
            px = np.hstack([p[:m], p[(m + 3*image):(m + 3*(image+1))]])
            e = np.zeros(m+3)
            e[m+j] = epsilon
            Ji[:,j] = (residualsfun(px+e, image) - residualsfun(px-e, image)) / (2*epsilon)
        J[image] = Ji

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

    return p


# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.
def levenberg_marquardt(residualsfun, p0, printbool=False, stop_precision = 1e-3, num_iterations=100, finite_difference_epsilon=1e-5):

    #p = [0,0,0] #task 1.5
    
    p = p0.copy()
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

        #delta = -np.linalg.inv(JTJ + mu * np.eye(dim)) @ JTr
        delta = np.linalg.solve(JTJ + mu*np.eye(dim), -JTr)

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


def levenberg_marquardt_test(residualsfun, p0, num_iterations=1000, finite_difference_epsilon=1e-5):
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




def levenberg_marquardt2(residualsfun, residualsfun_individual, p0, printbool=False, stop_precision = 1e-3, num_iterations=100, finite_difference_epsilon=1e-5):

    print("innit lM2")
    #p = [0,0,0] #task 1.5

    l = 351
    m = 5 + 7*3
    n = 7 

    p = p0
    e = finite_difference_epsilon

    J1 = jacobian2(residualsfun,p,e,l,m,n)
    J2 = jacobian3(residualsfun_individual,p,e,l,m,n)
    J2 = block_diag(*J2)

    # Forming A11 -- A22 from Figure 5 in assignment
    A11 = J1.T @ J1
    A12 = J1.T @ J2
    A21 = A12.T
    A22 = J2.T @ J2

    mu = np.max((np.max(np.diag(A11)), np.max(np.diag(A22)))) * 1e-3
   

    print("started")

    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.

        J1 = jacobian2(residualsfun,p,e,l,m,n)

        J2 = jacobian3(residualsfun_individual,p,e,l,m,n)


        J2 = block_diag(*J2)


        # Forming A11 -- A22 from Figure 5 in assignment

        A11 = J1.T @ J1
        A12 = J1.T @ J2
        A21 = A12.T
        A22 = J2.T @ J2

        A11 += mu * np.eye(m)
        A22 += mu * np.eye(3*l)

        A22_inv = np.linalg.inv(A22)

        d1 = -(J1.T @ residualsfun(p))
        d2 = -(J2.T @ residualsfun(p))

        delta_stat = np.linalg.solve(A11 + mu * np.eye(m) - A12@A22_inv@A21, d1-A12@A22_inv@d2)
        delta_dyn = np.linalg.solve(A22 + mu * np.eye(3*l), d2 - A21 @ delta_stat)

        delta = np.hstack((delta_stat,delta_dyn))

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
