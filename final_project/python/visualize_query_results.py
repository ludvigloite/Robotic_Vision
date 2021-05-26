import matplotlib.pyplot as plt
import numpy as np
from util import *

# This script uses example data. You will have to modify the
# loading code below to suit how you structure your data.

def visualize_query_results(imageName, runOwnImages):

    if runOwnImages:
        model = 'storedData/model'
        query = f'storedData/query/{imageName}'
        K       = np.loadtxt(f'{model}/K.txt')       # Intrinsic matrix.
        X       = np.load(f'{model}/X.npy', allow_pickle=True)       # 3D points [shape: 4 x num_points].
        T_m2q   = np.loadtxt(f'{query}_T_m2q.txt')   # Model-to-query transformation (produced by your localization script).
        matches = np.loadtxt(f'{query}_matches.txt') # Initial 2D-3D matches (see usage code below).
        inliers = np.loadtxt(f'{query}_inliers.txt') # Indices of inlier matches (see usage code below).
        u       = np.loadtxt(f'{query}_u.txt')       # Image location of features detected in query image (produced by your localization script).
        I       = plt.imread(f'{query}.jpg')         # Query image.

    else:
        model = '../visualization_sample_data'
        query = f'../visualization_sample_data/query/{imageName}'
        K       = np.loadtxt(f'{model}/K.txt')       # Intrinsic matrix.
        X       = np.loadtxt(f'{model}/X.txt')       # 3D points [shape: 4 x num_points].
        T_m2q   = np.loadtxt(f'{query}_T_m2q.txt')   # Model-to-query transformation (produced by your localization script).
        matches = np.loadtxt(f'{query}_matches.txt') # Initial 2D-3D matches (see usage code below).
        inliers = np.loadtxt(f'{query}_inliers.txt') # Indices of inlier matches (see usage code below).
        u       = np.loadtxt(f'{query}_u.txt')       # Image location of features detected in query image (produced by your localization script).
        I       = plt.imread(f'{query}.jpg')         # Query image.


    assert X.shape[0] == 4
    assert u.shape[0] == 2

    # If you have colors for your point cloud model, then you can use this.
    #c = np.loadtxt(f'{model}/c.txt') # RGB colors [shape: num_points x 3].
    # Otherwise you can use this, which colors the points according to their Y.
    c = None

    # txt does not save datatype, so we do a conversion here.
    matches = matches.astype(np.int32)
    inliers = inliers.astype(np.int32)

    # These control the location and the viewing target
    # of the virtual figure camera, in the two views.
    # You will probably need to change these to work
    # with your scene.
    lookfrom1 = np.array((0,-20,5))
    lookat1   = np.array((0,0,6))
    lookfrom2 = np.array((25,-5,10))
    lookat2   = np.array((0,0,10))

    # 'matches' is assumed to be a Nx2 array, where the
    # first column is the index of the 2D point in the
    # query image and the second column is the index of
    # its matched 3D point.
    assert matches.shape[1] == 2
    u_matches = u[:,matches[:,0]]
    X_matches = X[:,matches[:,1]]

    # 'inliers' is assumed to be a 1D array of indices
    # of the good matches, e.g. as identified by your
    # PnP+RANSAC strategy.
    u_inliers = u_matches[:,inliers]
    X_inliers = X_matches[:,inliers]

    u_hat = project(K, T_m2q@X_inliers)
    e = np.linalg.norm(u_hat - u_inliers, axis=0)

    fig = plt.figure(figsize=(10,8))

    plt.subplot(221)
    plt.imshow(I)
    plt.scatter(*u_hat, marker='+', c=e)
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.colorbar(label='Reprojection error (pixels)')
    plt.title('Query image and reprojected points')

    plt.subplot(222)
    plt.hist(e, bins=50)
    plt.xlabel('Reprojection error (pixels)')

    plt.subplot(223)
    draw_model_and_query_pose(X, T_m2q, K, lookat1, lookfrom1, c=c)
    plt.title('Model and localized pose (top view)')

    plt.subplot(224)
    draw_model_and_query_pose(X, T_m2q, K, lookat2, lookfrom2, c=c)
    plt.title('Model and localized pose (side view)')

    plt.tight_layout()
    
    if imageName[:10]=='calibrated':
        plt.savefig(f'results/visualiseQuery_calibrated_{imageName[11:]}.png')
    else:
        plt.savefig(f'results/visualiseQuery_{imageName}.png')

    print(f"Saved file to results/visualiseQuery_{imageName}.png")
    #plt.show()
