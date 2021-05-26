import numpy as np
import cv2 as cv
import shutil
import random

import pydegensac

from task45_model import *
from visualize_query_results import *

from util import * 
from LM import jacobian_single_image

def localize(model, KPath, imageName, runOwnImages):
    if runOwnImages:
        queryImage = f"model_images/{imageName}.jpg"
    else:
        queryImage = f"../hw5_data_ext/{imageName}.JPG"
        
    queryImg = cv.imread(queryImage)
    X = model.X
    modelDesc = model.descriptor
    kp1 = model.kp1
    K = np.loadtxt(KPath)

    #legg til keypoint også

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    qkp, qDes = sift.detectAndCompute(queryImg,None)

    # BFMatcher with default params and k best matches
    bf = cv.BFMatcher()
    matches = bf.knnMatch(qDes, modelDesc, k=2)

    # Apply ratio test (see D.Lowe paper and opencv docs)
    good = []
    good2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance: # 0.75
            good.append([m])
            good2.append(m)


    """
    th = 0.5
    n_iter = 50000

    t=time()
    cv_H, cv_mask = verify_cv2_fundam(kp1, qkp, good2, th, n_iter )
    print ("{0:.5f}".format(time()-t), ' sec cv2')

    t=time()
    cmp_H, cmp_mask = verify_pydegensac_fundam(kp1,qkp,good2, th, n_iter)
    print ("{0:.5f}".format(time()-t), ' sec pydegensac')

    #draw_matches_F(kp1, kp2, good2, self.img1, self.img2, cv_mask, "cv")
    #draw_matches_F(kp1, kp2, good2, self.img1, self.img2, cmp_mask,"pydegensac")

    degensac_mask = cmp_mask
    cv_mask = cv_mask
    """

    
    
    
    u = np.array([qkp[i].pt for i in range(len(qkp))]).T
    matchesIndexes = np.array([])
    for i in range(np.array(good).size):
        matchesIndexes = np.append(matchesIndexes, np.array([good[i][0].queryIdx, good[i][0].trainIdx]))
    matchesIndexes = matchesIndexes.reshape([len(good),2]).astype(int)

    X = (X.T[matchesIndexes[:, 1]]).T
    X_dehom = (X.T[:,:3]).T

    _, rotVec, transVec, inliers = cv.solvePnPRansac(X_dehom.T, u[:, matchesIndexes[:, 0]].T, K, None)

    #inliers = cv_mask
    
    inliers = inliers[:,0]
    T = np.hstack([cv.Rodrigues(rotVec)[0],transVec])
    T_homo = np.vstack([T,[0,0,0,1]]) #homogenizing

    uv2 = u[:, matchesIndexes[:, 0]]
    uv2 = uv2[:,inliers]
    X = X[:,inliers]

    #### Checking reprojection error BEFORE bundle adjustment
    X2 = T @ X
    uv2_proj = project(K,X2)
    r2 = np.mean(np.sqrt((uv2[0,:] - uv2_proj[0,:])**2 + (uv2[1,:]- uv2_proj[1,:]) ** 2))
    print(f"Before Bundle Adjustment:\nMean reprojection error query_img: {r2}")


    ########### Bundle adjustment ###########

    params0 = np.zeros(6)
    LS_result = scipy.optimize.least_squares(fun=residuals_localize,x0=params0, verbose=1, args=(X,T, uv2, K))
    params = LS_result.x
    T_query = compose_T2(params, T)

    #### Checking reprojection error AFTER bundle adjustment
    X_query = T_query @ X
    uv2_proj = project(K,X_query)
    r2 = np.mean(np.sqrt((uv2[0,:] - uv2_proj[0,:])**2 + (uv2[1,:]- uv2_proj[1,:]) ** 2))
    print(f"After Bundle Adjustment:\nMean reprojection error query_img: {r2}")

    T = T_query


    saveQueryParameters(T,matchesIndexes,inliers,u,imageName, queryImage)


def saveQueryParameters(T,matches,inliers,u,imageName, queryImage):

    np.savetxt(f"storedData/query/{imageName}_T_m2q.txt", T)           # Model-to-query transformation (produced by your localization script).
    np.savetxt(f"storedData/query/{imageName}_matches.txt", matches)   # Initial 2D-3D matches (see usage code below).
    np.savetxt(f"storedData/query/{imageName}_inliers.txt", inliers)   # Indices of inlier matches (see usage code below).
    np.savetxt(f"storedData/query/{imageName}_u.txt", u)               # Image location of features detected in query image (produced by your localization script).
    if imageName[:10]=='calibrated':
        shutil.copy(queryImage, "storedData/query/calibrated")
    else:
        shutil.copy(queryImage, "storedData/query/")
    
    print("Query parameters saved")


def estimateCovariance(imageName, nrOfSamples, model, dataPath="storedData/", printBool=True):
    queryImage = f"../hw5_data_ext/{imageName}.JPG"
    queryImg = cv.imread(queryImage)

    X = np.load(dataPath+f"/model/X.npy")
    K_ = np.load(dataPath+f"/model/K.npy")
    modelDesc = np.load(dataPath+f"/model/descriptor.npy")
    matches = np.loadtxt(dataPath+f"/query/{imageName}_matches.txt").astype(int)



    # Initiate SIFT detector
    sift = cv.SIFT_create()
    qkp, qDes = sift.detectAndCompute(queryImg,None)

    # BFMatcher with default params and k best matches
    bf = cv.BFMatcher()
    matches = bf.knnMatch(qDes, modelDesc, k=2)

    # Apply ratio test (see D.Lowe paper and opencv docs)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance: # 0.75
            good.append([m])
    
    
    u = np.array([qkp[i].pt for i in range(len(qkp))]).T
    matchesIndexes = np.array([])
    for i in range(np.array(good).size):
        matchesIndexes = np.append(matchesIndexes, np.array([good[i][0].queryIdx, good[i][0].trainIdx]))
    matchesIndexes = matchesIndexes.reshape([len(good),2]).astype(int)

    X_dehom = (X.T[:,:3][matchesIndexes[:, 1]]).T

    paramsAll = np.empty([6, nrOfSamples])



    task = "c"
    for i in range(nrOfSamples):
        if task == "a":
            # Task 3.4 a)
            eta_f = random.gauss(0, 50)
            eta_cx = random.gauss(0, 0.1)
            eta_cy = random.gauss(0, 0.1)
        if task == "b":
            # Task 3.4 b)
            eta_f = random.gauss(0, 0.1)
            eta_cx = random.gauss(0, 50)
            eta_cy = random.gauss(0, 0.1)
        if task == "c":
            # Task 3.4 c)
            eta_f = random.gauss(0, 0.1)
            eta_cx = random.gauss(0, 0.1)
            eta_cy = random.gauss(0, 50)

        K_cov = np.array([  [eta_f, 0, eta_cx],
                            [0, eta_f, eta_cy],
                            [0, 0, 0]])

        K = K_ + K_cov

        _, rotVec, transVec, inliers = cv.solvePnPRansac(X_dehom.T, u[:, matchesIndexes[:, 0]].T, K, None)
        
        inliers = inliers[:,0]
        T = np.hstack([cv.Rodrigues(rotVec)[0],transVec])
        T = np.vstack([T,[0,0,0,1]])

        paramsAll[:, i] = decompose_T(T)

    cov = np.cov(paramsAll)

    if printBool:
        angles = np.sqrt(np.diag(cov)[:3])
        positions = np.sqrt(np.diag(cov)[3:]) * 1000
        print('\n\nStandard deviations - Estimated')
        print('\nRotation vector')
        print('----------------------------')
        print(f'Angle x: {np.round(angles[0],4)} degrees')
        print(f'Angle y: {np.round(angles[1],4)} degrees')
        print(f'Angle z: {np.round(angles[2],4)} degrees')
        print('\nTranslation Vector')
        print('----------------------------')
        print(f'X: {np.round(positions[0],4)} mm')
        print(f'Y: {np.round(positions[1],4)} mm')
        print(f'Z: {np.round(positions[2],4)} mm\n')

    return cov

        


def calculateQueryCovariance(queryImage, sigma, dataPath="storedData/", printBool=True):
    queryImg = cv.imread(queryImage)

    X = np.load(dataPath+f"/model/X.npy")
    K = np.load(dataPath+f"/model/K.npy")
    uv1 = np.loadtxt(dataPath+"/model/allMatches_img1.txt")
    uv2 = np.loadtxt(dataPath+f"/query/{queryImage}_u.txt")
    T = np.loadtxt(dataPath+f"/query/{queryImage}_T_m2q.txt")
    inliers = np.loadtxt(dataPath+f"/query/{queryImage}_inliers.txt").astype(int)
    matches = np.loadtxt(dataPath+f"/query/{queryImage}_matches.txt").astype(int)

    X = X[:, matches[:,1]]
    X_inliers = X[:,inliers]
    u_matches = uv2[:,matches[:,0]]
    u_inliers = u_matches[:,inliers]

    nrOfPoints = X_inliers.shape[1]

    params = decompose_T(T)
    params = np.zeros(6)

    residualsfun = lambda params : residuals_single_image(u_inliers, K, T, X_inliers, params)

    epsilon = 1e-5
    J = jacobian_covariance(residualsfun, params, epsilon, 6)
    Epsilon_r = sigma * np.eye(int(2*nrOfPoints))
    Epsilon_p = np.linalg.inv(J.T @ np.linalg.inv(Epsilon_r) @ J)

    # rVec_std = np.rad2deg(np.sqrt(np.diag(Epsilon_p))[:3])
    rVec_std = np.sqrt(np.diag(Epsilon_p))[:3]
    tVec_std = np.sqrt(np.diag(Epsilon_p))[3:]*1000
    

    if printBool:
        print('\n\nStandard deviations - Calculated')
        print('\nRotation vector')
        print('----------------------------')
        print(f'Angle X: {np.round(rVec_std[0],4)} degrees')
        print(f'Angle Y: {np.round(rVec_std[1],4)} degrees')
        print(f'Angle Z: {np.round(rVec_std[2],4)} degrees')
        print('\nTranslation Vector')
        print('----------------------------')
        print(f'X: {np.round(tVec_std[0],4)} mm')
        print(f'Y: {np.round(tVec_std[1],4)} mm')
        print(f'Z: {np.round(tVec_std[2],4)} mm\n')
    

    return Epsilon_p

if __name__ == "__main__":

    useCalibrated = False
    runOwnImages = True
    useSavedModel = False

    whichRansac = "degensac"

    if runOwnImages:
        localizeImg = "IMG_3896"

        if useCalibrated:
            m2 = model("model_images/calibrated/IMG_3898.jpg", "model_images/calibrated/IMG_3899.jpg", "calibration/results/calibrated_K.txt",2.06)
            localizeImg = f"calibrated/{localizeImg}"
        else:
            m2 = model("model_images/IMG_3898.jpg", "model_images/IMG_3899.jpg", "calibration/results/calibrated_K.txt",2.05)

        if useSavedModel:
            m2.loadModel("storedData/model") 

        else:
            m2.performMatching()
            m2.generateStructure(whichRansac=whichRansac)
            m2.saveModel()

        localize(m2, "calibration/results/calibrated_K.txt", localizeImg, runOwnImages=True)
        visualize_query_results(localizeImg, runOwnImages=True)

        calculateQueryCovariance(localizeImg, 1)

    else:
        localizeImg = "IMG_8210"

        m = model("../hw5_data_ext/IMG_8207.JPG", "../hw5_data_ext/IMG_8229.JPG", "../hw5_data_ext/K.txt", 6.2)

        if useSavedModel:
            m.loadModel("storedData/model") 

        else:
            m.performMatching()
            m.generateStructure(whichRansac=whichRansac)
            m.saveModel()

        estimateCovariance(localizeImg, 500, m, dataPath="storedData/", printBool=True)

        localize(m, "../hw5_data_ext/K.txt", localizeImg, runOwnImages=False)
        visualize_query_results(localizeImg, runOwnImages=False)

        calculateQueryCovariance(localizeImg, 1)


