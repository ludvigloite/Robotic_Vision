import matplotlib.pyplot as plt
import numpy as np

def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def draw_frame(K, T, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X)
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis


# From assignment 4

def estimate_H(xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.

    n = XY.shape[1]

    A = np.vstack((np.array([XY[0], XY[1], np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), -XY[0]*xy[0], -XY[1]*xy[0], -xy[0]]).T,
                    np.array([np.zeros(n), np.zeros(n), np.zeros(n), XY[0], XY[1], np.ones(n), -XY[0]*xy[1], -XY[1]*xy[1], -xy[1]]).T))

    _,_,VT = np.linalg.svd(A)
    H = VT.T[:,-1].reshape((3,3))

    return H

def decompose_H(H):
    # Tip: Use np.linalg.norm to compute the Euclidean length
    T1 = np.eye(4)
    T2 = np.eye(4)

    k1 = np.linalg.norm(H[:,0])
    k2 = -k1

    r1 = H[:,0] / k1
    r2 = H[:,1] / k1
    r3 = np.cross(r1, r2)
    t = H[:,2] / k1

    T1[:3,:4] = np.column_stack((r1, r2, r3, t))
    T1[:3,:3] = closest_rotation_matrix(np.column_stack((r1, r2, r3)))
    
    r1 = H[:,0] / k2
    r2 = H[:,1] / k2
    r3 = np.cross(r1, r2)
    t = H[:,2] / k2

    T2[:3,:4] = np.column_stack((r1, r2, r3, t))
    T2[:3,:3] = closest_rotation_matrix(np.column_stack((r1, r2, r3)))

    if T1[2,3] >= 0:
        T = T1
    else:
        T = T2

    return T

def closest_rotation_matrix(Q):
    U,_,V = np.linalg.svd(Q)
    R = U @ V
    # print(np.linalg.norm(R @ R.T - np.eye(3)))
    # print(np.linalg.norm(Q @ Q.T - np.eye(3)))
    return R
