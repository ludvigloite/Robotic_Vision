import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def estimate_H(xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.

    n = XY.shape[1]

    A = np.zeros((2*n, 9))
    A[0::2] = np.array([XY[0], XY[1], np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), -XY[0]*xy[0], -XY[1]*xy[0], -xy[0]]).T
    A[1::2] = np.array([np.zeros(n), np.zeros(n), np.zeros(n), XY[0], XY[1], np.ones(n), -XY[0]*xy[1], -XY[1]*xy[1], -xy[1]]).T

    U, s, VT = np.linalg.svd(A)
    h = VT.T[:,-1]

    H = h.reshape((3,3))
    return H

def calc_reprojection_error(uv, uv_predicted):
    reprojection_err = np.sqrt(np.diag((uv.T - uv_predicted.T) @ (uv - uv_predicted)))

    avg_rep_err = np.average(reprojection_err)
    max_rep_err = np.max(reprojection_err)
    min_rep_err = np.min(reprojection_err)
    return [avg_rep_err, max_rep_err, min_rep_err]

def decompose_H(H):
    # Tip: Use np.linalg.norm to compute the Euclidean length

    T1 = np.eye(4) # Placeholder, replace with your implementation
    T2 = np.eye(4) # Placeholder, replace with your implementation

    k_abs = np.linalg.norm(H[:,0])

    H1 = 1/k_abs * H
    H2 = -1/k_abs * H

    T1[:3, :2] = H1[:, :2]
    T1[:3, 2] = np.cross(H1[:, 0], H1[: ,1])
    T1[:3, 3] = H1[:, 2]

    T2[:3, :2] = H2[:, :2]
    T2[:3, 2] = np.cross(H2[:, 0], H2[: ,1])
    T2[:3, 3] = H2[:, 2]

    T1z = T1[2, 3]
    T2z = T2[2, 3]
    
    if T1z > 0:
        T = T1
    elif T2z > 0:
        T = T2
    else:
        print("Both poses are invalid!")
        T = None
        return T, None
    
    # T = T2

    Q = np.copy(T[:3,:3])
    R = closest_rotation_matrix(Q)
    T[:3, :3] = np.copy(R)

    return T

def closest_rotation_matrix(Q):
    U, s, VT = np.linalg.svd(Q)
    R = U @ VT
    return R

def project(K, X):
    """
    Computes the pinhole projection of an (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the dehomogenized pixel
    coordinates as an array of size 2xN.
    """
    if len(X.shape) == 2:
        X = X[None]

    uvw = K@X[:,:3,:]
    # uvw /= uvw[2,:]
    uvw = np.array([uvw[i]/uvw[i,2,:] for i in range(X.shape[0])])
    # return uvw[:2,:]
    return uvw[:,:2,:]

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

def generate_figure(fig, image_number, K, T, uv, uv_predicted, XY):

    fig.suptitle('Image number %d' % image_number)

    #
    # Visualize reprojected markers and estimated object coordinate frame
    #
    I = plt.imread('quanser_sequence/video%04d.jpg' % image_number)
    plt.subplot(121)
    plt.imshow(I)
    # draw_frame(K, T, scale=4.5)
    plt.scatter(uv[0,:], uv[1,:], color='red', label='Detected')
    plt.scatter(uv_predicted[0,:], uv_predicted[1,:], marker='+', color='yellow', label='Predicted')
    plt.legend()
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])

    #
    # Visualize scene in 3D
    #
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.plot(XY[0,:], XY[1,:], np.zeros(XY.shape[1]), '.') # Draw markers in 3D
    # pO = np.linalg.inv(T)@np.array([0,0,0,1]) # Compute camera origin
    # pX = np.linalg.inv(T)@np.array([6,0,0,1]) # Compute camera X-axis
    # pY = np.linalg.inv(T)@np.array([0,6,0,1]) # Compute camera Y-axis
    # pZ = np.linalg.inv(T)@np.array([0,0,6,1]) # Compute camera Z-axis
    # plt.plot([pO[0], pZ[0]], [pO[1], pZ[1]], [pO[2], pZ[2]], color='blue') # Draw camera Z-axis
    # plt.plot([pO[0], pY[0]], [pO[1], pY[1]], [pO[2], pY[2]], color='green') # Draw camera Y-axis
    # plt.plot([pO[0], pX[0]], [pO[1], pX[1]], [pO[2], pX[2]], color='red') # Draw camera X-axis
    # ax.set_xlim([-40, 40])
    # ax.set_ylim([-40, 40])
    # ax.set_zlim([-25, 25])
    # ax.set_xlabel('X')
    # ax.set_zlabel('Y')
    # ax.set_ylabel('Z')

    plt.tight_layout()
