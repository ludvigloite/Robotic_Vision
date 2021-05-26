import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from hw5.triangulate_many import *

def decompose_T(T):
    R = T[:3, :3]

    r = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

    tx = T[0, 3]
    ty = T[1, 3]
    tz = T[2, 3]
    return np.array([r[0], r[1], r[2], tx, ty, tz])

def compose_T(params):
    r = params[:3]
    tx = params[3]
    ty = params[4]
    tz = params[5]
    R = Rotation.from_euler('xyz', r, degrees=True).as_matrix()

    T = np.zeros([4,4])
    T[:3,:3] = R
    T[:3,3] = [tx, ty, tz]
    T[3,3] = 1
    
    return T

def compose_T2(params, T_in):
    R = rotate_z(params[0]) @ rotate_y(params[1]) @ rotate_x(params[2])
    R = closest_rotation_matrix(R)

    T = np.zeros([4,4])
    T[:3,:3] = R @ T_in[:3,:3]
    T[:3,3] = T_in[:3,3] + [params[3], params[4], params[5]]
    T[3,3] = 1
    
    return T

def closest_rotation_matrix(Q):
    U,_,V = np.linalg.svd(Q)
    R = U @ V
    return R

def residuals2(params, X, T, uv_train, uv_query, K, nrOf3DPoints):
    cam_params = params[:6]
    X_params = params[6:]

    X_points = X + np.vstack((X_params.reshape((nrOf3DPoints,3)).T, np.zeros(X.shape[1])))
    T_query = compose_T2(cam_params, T)

    X_train = X_points
    X_query = T_query @ X_points

    uv_train_proj = project(K,X_train)
    uv_query_proj = project(K,X_query)

    r_uv_train = uv_train[:2,:] - uv_train_proj
    r_uv_query = uv_query[:2,:] - uv_query_proj

    r_uv_train = r_uv_train.T.reshape(1, 2*np.size(r_uv_train,1))
    r_uv_query = r_uv_query.T.reshape(1, 2*np.size(r_uv_query,1))

    r = np.append(r_uv_query, r_uv_train)
    return r



def residuals_localize(params, X, T, uv_query, K):
    cam_params = params

    T_query = compose_T2(cam_params, T)

    X_query = T_query @ X

    uv_query_proj = project(K,X_query)
    r_uv_query = uv_query[:2,:] - uv_query_proj
    r_uv_query = r_uv_query.T.reshape(1, 2*r_uv_query.shape[1])

    return r_uv_query[0]

def residuals_localize_weighted(params, X, T, uv_query, K):
    cam_params = params

    T_query = compose_T2(cam_params, T)

    X_query = T_query @ X

    uv_query_proj = project(K,X_query)
    r_uv_query = uv_query[:2,:] - uv_query_proj

    task = "3.3"
    if task == "3.2":
        # Task 3.2
        sigma_u = 1
        sigma_v = 1
    elif task == "3.3":
        # Task 3.3
        sigma_u = 50#**2
        sigma_v = 0.1#**2
    weights = calculateWeights(sigma_u, sigma_v, r_uv_query.shape[1])
    r_uv_query = weights @ r_uv_query.T.ravel()

    # r_uv_query = r_uv_query.T.reshape(1, 2*np.size(r_uv_query,1))

    return r_uv_query


def residuals_localize_weighted_4_1(params, X, T, uv_query, K, sigma_u, sigma_v, sigma_f):
    cam_params = params

    T_query = compose_T2(cam_params, T)

    X_query = T_query @ X

    uv_query_proj = project(K,X_query)
    r_uv_query = uv_query[:2,:] - uv_query_proj

    weights = np.zeros([X_query.shape[1]*2,X_query.shape[1]*2])

    for i in range(X_query.shape[1]):
        """
        u_hat_std = np.sqrt(sigma_u**2+((X_query[0,i])**2)/((X_query[2,i])**2)*sigma_f**2)
        v_hat_std = np.sqrt(sigma_v**2+((X_query[1,i])**2)/((X_query[2,i])**2)*sigma_f**2)
        r_uv_query[0,i] = r_uv_query[0,i]/u_hat_std
        r_uv_query[1,i] = r_uv_query[1,i]/v_hat_std
        """
        u_hat_std = np.sqrt(sigma_u**2+((X_query[0,i])**2)/((X_query[2,i])**2)*sigma_f**2)
        v_hat_std = np.sqrt(sigma_v**2+((X_query[1,i])**2)/((X_query[2,i])**2)*sigma_f**2)
        weights[2*i,2*i] = 1/u_hat_std
        weights[2*i+1,2*i+1] = 1/v_hat_std

    r_uv_query = weights @ r_uv_query.T.ravel()

    return r_uv_query

    #r_uv_query = r_uv_query.T.reshape(1, 2*r_uv_query.shape[1])

    #return r_uv_query[0]


def residuals(params0, uv1, uv2, nrOfGlobalParams, K, params):
    '''
    Calculate residuals for image 1 and image 2 using estimated 3D positions and camera parameters.
    params can either contain exactly one 3D point, or all matched 3D points.
    In:
        params: [X, Y, Z, idx, cameraparams] or [X, Y, Z, cameraparams]
    Out:
        res: [x11, y11, x12, y12, ..., x21, y21, x22, y22]
        xij, where i: 3D-point correspondence, and j: image index
    '''

    nrOf3DPoints = int((params.shape[0] - nrOfGlobalParams)/3)

    #x_prediction = params[6:].reshape((nrOf3DPoints,3)).T
    
    cam2_old = params[:nrOfGlobalParams]
    cam2_new = params0[:nrOfGlobalParams]
    cam2 = cam2_old+cam2_new
    #opt_cam2 = params[:len(cam2)]
    X_old = params[nrOfGlobalParams:].reshape(nrOf3DPoints,3)
    X_new = params0[nrOfGlobalParams:].reshape(nrOf3DPoints,3)
    X = X_old+X_new

    #X = params[:nrOf3DPoints*3].reshape(nrOf3DPoints, 3)

    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2 = compose_T(cam2)[:3,:]

    X1 = P1@to_homogeneous(X.T)
    X2 = P2@to_homogeneous(X)

    # uv1_hat = project(K, X.T) # Allmost no difference, since P1 should not change that much
    uv1_hat = project(K, X1)
    uv2_hat = project(K, X2)

    res = np.array([np.array([(uv1_hat.T - uv1[:2].T)[i], (uv2_hat.T - uv2[:2].T)[i]]).ravel() for i in range(nrOf3DPoints)]).ravel()
    return res


def jacobian_covariance(residualsfun, params, epsilon, nrOfGlobalParams):
    n = residualsfun(params).shape[0]
    m = params.shape[0]
    
    J = np.zeros([n,m])
    
    for i in range(nrOfGlobalParams):
        e = np.zeros(nrOfGlobalParams)
        e[i] = epsilon
        J[:,i] = (residualsfun(params+e)-residualsfun(params-e)) / (2*epsilon)
    return J

def residuals_single_image(uv, K, T, X, params):
    '''
    Calculate residuals for image 1 and image 2 using estimated 3D positions and camera parameters.
    params can either contain exactly one 3D point, or all matched 3D points.
    In:
        params: [X, Y, Z, idx, cameraparams] or [X, Y, Z, cameraparams]
    Out:
        res: [x11, y11, x12, y12, ..., x21, y21, x22, y22]
        xij, where i: 3D-point correspondence, and j: image index
    '''
    nrOfGlobalParams = params.shape[0]
    cam = params

    R1 = np.diag(np.zeros(3))
    R1 = T[:3,:3]


    R1 = rotate_z(cam[2]) @ rotate_y(cam[1]) @ rotate_x(cam[0]) @ R1
    T[:3,:3] = R1[:3,:3]
    T[:3, 3] += cam[3:]

    # P = compose_T(cam)[:3,:]
    P = T[:3,:]
    X1 = P@X

    uv_hat = project(K, X1)
    nrOfPoints = uv_hat.shape[1]
    
    # res = np.array([np.array([uv_hat[0][i] - uv[0][i], uv_hat[1][i] - uv[1][i]]).ravel() for i in range(nrOfPoints)]).ravel()
    # Task 3.3
    sigma_u = 50**2
    sigma_v = 0.1**2
    weights = calculateWeights(sigma_u, sigma_v, nrOfPoints)

    res = np.array([np.array([uv_hat[0][i] - uv[0][i], uv_hat[1][i] - uv[1][i]]).ravel() for i in range(nrOfPoints)]).ravel()
    res = weights @ res
    
    return res

def calculateWeights(sigma_u, sigma_v, n_points):
    # sigma_u = 50**2
    # sigma_v = 0.1**2
    Epsilon_r = np.zeros((n_points*2,n_points*2))
    idx_u = np.array([i for i in range(n_points*2) if i % 2 != 0])
    idx_v = np.array([i for i in range(n_points*2) if i % 2 == 0])
    Epsilon_r[idx_u,idx_u] = np.ones(n_points) * sigma_u**2
    Epsilon_r[idx_v,idx_v] = np.ones(n_points) * sigma_v**2

    w = np.linalg.inv(np.linalg.cholesky(Epsilon_r))
    return w

def to_homogeneous(X):
    (n, m) = X.shape
    temp = np.ones([n+1,m])
    temp[:n,:] = np.copy(X)
    return temp


def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]])

def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def project(K, X):
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def project_camera_frame(K, T, scale):
    """
    Draw the axes of T and a pyramid, representing the camera.
    """
    s = scale
    X = []
    X.append(np.array([0,0,0,1]))
    X.append(np.array([-s,-s,1.5*s,1]))
    X.append(np.array([+s,-s,1.5*s,1]))
    X.append(np.array([+s,+s,1.5*s,1]))
    X.append(np.array([-s,+s,1.5*s,1]))
    X.append(np.array([5.0*s,0,0,1]))
    X.append(np.array([0,5.0*s,0,1]))
    X.append(np.array([0,0,5.0*s,1]))
    X = np.array(X).T
    u,v = project(K, T@X)
    lines = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
    plt.plot([u[0], u[5]], [v[0], v[5]], color='#ff5555', linewidth=2)
    plt.plot([u[0], u[6]], [v[0], v[6]], color='#33cc55', linewidth=2)
    plt.plot([u[0], u[7]], [v[0], v[7]], color='#44aaff', linewidth=2)
    for (i,j) in lines:
        plt.plot([u[i], u[j]], [v[i], v[j]], color='black')

def draw_model_and_query_pose(X, T_m2q, K,
    lookat=np.array((0.0, 0.0, 0.0)),
    lookfrom=np.array((0.0, 0.0, -15.0)),
    point_size=10,
    frame_size=0.5,
    c=None):

    """
              X: Point cloud model of [shape (3 or 4)xN].
          T_m2q: Transformation from model to query camera coordinates (e.g. as obtained from OpenCV's solvePnP).
              K: Intrinsic matrix for the virtual 'figure camera'.
    lookat|from: The viewing target and origin of the virtual figure camera.
     point_size: Radius of a point (in pixels) that is 1 unit away. (Points further away will appear smaller.)
     frame_size: The length (in model units) of the camera and coordinate frame axes.
              c: Color associated with each point in X [shape Nx3].
    """

    assert X.ndim == 2, 'X must be a (3 or 4)xN array'
    assert X.shape[1] > 0, 'X must have at least one point'

    X = X.copy()
    if X.shape[0] == 3:
        X = np.vstack([X, np.ones_like(X[0,:])])
    else:
        X = X/X[3,:]

    if c is None:
        c = X[1,:]
    else:
        c = c.copy()
        if np.max(c) > 1.0:
            c = c / 256.0

    # Create transformation from model to 'figure camera'
    T_f2m = np.eye(4)
    T_f2m[:3,2] = (lookat - lookfrom)
    T_f2m[:3,2] /= np.linalg.norm(T_f2m[:3,2])
    T_f2m[:3,0] = np.cross(np.array((0,1,0)), T_f2m[:3,2])
    T_f2m[:3,0] /= np.linalg.norm(T_f2m[:3,0])
    T_f2m[:3,1] = np.cross(T_f2m[:3,2], T_f2m[:3,0])
    T_f2m[:3,1] /= np.linalg.norm(T_f2m[:3,1])
    T_f2m[:3,3] = lookfrom
    T_m2f = np.linalg.inv(T_f2m)

    # Transform point cloud model into 'figure camera' frame
    X_f = T_m2f@X
    visible = X_f[2,:] > 0
    not_visible = X_f[2,:] < 0
    X_f = X_f[:,visible]
    if np.sum(not_visible) > np.sum(visible):
        print('[draw_model_and_query_pose] Most of point cloud is behind camera, is that intentional?')

    # Project point cloud with depth sorting
    T_q2m = np.linalg.inv(T_m2q)
    u = K@X_f[:3,:]
    u = u[:2,:]/u[2,:]
    i = np.argsort(-X_f[2,:])
    plt.scatter(*u[:,i], c=c[i], s=(point_size**2)/X_f[2,:], rasterized=True)

    # Project the coordinate frame axes of the localized query camera
    project_camera_frame(K, T_m2f@T_q2m, scale=frame_size)

    plt.axis('image')

    # Arbitrarily set axis limits to 2*principal point
    scale = 2

    #plt.xlim([0.0, K[0,2]*scale])
    #plt.ylim([K[1,2]*scale, 0.0])

    plt.xlim([500, 3000])
    plt.ylim([2500, 1000])
