import numpy as np
from util import *


def jacobian(residualsfun, params, epsilon, nrOfGlobalParams):
    n = residualsfun(params).shape[0]
    m = params.shape[0]
    measurements = int(n / 4)
    J = np.zeros([n,m])
    #  Create 3D points block
    for measIdx in range(measurements):
        M = np.zeros([4,3])
        # Iterate through X, Y and Z
        for i in range(3):
            e = np.zeros(4+nrOfGlobalParams)
            e[i] = epsilon
            p = np.hstack([params[measIdx*3:measIdx*3+3], measIdx, params[m-nrOfGlobalParams:]])
            M[:,i] = (residualsfun(p+e)-residualsfun(p-e)) / (2*epsilon)
        J[measIdx*4:(measIdx+1)*4, measIdx*3:(measIdx+1)*3] = np.copy(M)

    #  Create camera params blocks
    for i in range((m-nrOfGlobalParams), m):
        e = np.zeros(m)
        e[i] = epsilon
        J[:, i] = (residualsfun(params+e)-residualsfun(params-e)) / (2*epsilon)
    return J


def jacobian_single_image(residualsfun, params, epsilon, nrOfGlobalParams):
    n = residualsfun(params).shape[0]
    m = params.shape[0]
    
    J = np.zeros([n,m])
    
    for i in range(nrOfGlobalParams):
        e = np.zeros(nrOfGlobalParams)
        e[i] = epsilon
        J[:,i] = (residualsfun(params+e)-residualsfun(params-e)) / (2*epsilon)
    return J

def levenberg_marquardt(residualsfun, params, nrOfGlobalParams, printbool=False, stop_precision = 1e-6, num_iterations=10, finite_difference_epsilon=1e-5):    
    e = finite_difference_epsilon
    J = jacobian(residualsfun,params,e,nrOfGlobalParams)
    dim = J.shape[1]
    mu = np.max(np.diag((J.T @ J))) * 1e-3

    for iteration in range(num_iterations):

        J = jacobian(residualsfun,params,e,nrOfGlobalParams)

        if printbool:
            print("J= \n",J)
            print(f"JTJ = \n{J.T @ J}")
    
        
        JTJ = J.T @ J
        JTr = J.T @ residualsfun(params)

        # delta = -np.linalg.inv(JTJ + mu * np.eye(dim)) @ JTr
        delta = -np.linalg.inv(JTJ + mu * np.diag(np.diag(JTJ))) @ JTr # To enforce scale invariance (?)

        if printbool:
            print(f"delta  = {delta}")

        
        E_delta = np.sum(residualsfun(params+delta)**2)
        E = np.sum(residualsfun(params)**2)

        if E_delta < E:
            mu /= 3
            params += delta
        else:
            mu *= 2

        if np.linalg.norm(delta) < stop_precision:
            print("Stopped at {}".format(iteration))
            return params
    print("Failed at {}".format(iteration))

    return params
