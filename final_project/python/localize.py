import numpy as np
import cv2 as cv
import shutil
import random

from model import *
from visualize_query_results import *

from util import * 
from LM import jacobian_single_image

def localize(model, KPath, imageName, runOwnImages, useWeighted=False, saveParams=True, useRootSIFT=False, task41=False):
    if runOwnImages:
        queryImage = f"model_images/{imageName}.jpg"
    else:
        queryImage = f"../hw5_data_ext/{imageName}.JPG"
        
    queryImg = cv.imread(queryImage)
    X = model.X
    modelDesc = model.descriptor
    # K = np.loadtxt(KPath)
    K = model.K

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    qkp, qDes = sift.detectAndCompute(queryImg,None)

    if useRootSIFT:
        qDes /= (qDes.sum(axis=1, keepdims=True))
        qDes = np.sqrt(qDes)

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

    X = (X.T[matchesIndexes[:, 1]]).T
    X_dehom = (X.T[:,:3]).T

    _, rotVec, transVec, inliers = cv.solvePnPRansac(X_dehom.T, u[:, matchesIndexes[:, 0]].T, K, None)
    
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


    sigma_cx = 0.1
    sigma_cy = 50
    sigma_f = 0.1

    params0 = np.zeros(6)

    if useWeighted:
        LS_result = scipy.optimize.least_squares(fun=residuals_localize_weighted,x0=params0, verbose=2, args=(X,T, uv2, K))
    elif task41:
        LS_result = scipy.optimize.least_squares(fun=residuals_localize_weighted_4_1,x0=params0, verbose=2, args=(X,T, uv2, K, sigma_cx, sigma_cy, sigma_f))
    else:
        LS_result = scipy.optimize.least_squares(fun=residuals_localize,x0=params0, verbose=1, args=(X,T, uv2, K))
    params = LS_result.x
    J = LS_result.jac
    T_query = compose_T2(params, T)





    #### Checking reprojection error AFTER bundle adjustment
    X_query = T_query @ X

    checkStdDev4_1(J, X_query, sigma_cx, sigma_cy, sigma_f, withWeighting=False)


    uv2_proj = project(K,X_query)
    r2 = np.mean(np.sqrt((uv2[0,:] - uv2_proj[0,:])**2 + (uv2[1,:]- uv2_proj[1,:]) ** 2))
    print(f"After Bundle Adjustment:\nMean reprojection error query_img: {r2}")

    T = T_query

    if saveParams:
        saveQueryParameters(T,matchesIndexes,inliers,u,imageName, queryImage)

    return J


def checkStdDev4_1(J, X_query, sigma_u, sigma_v, sigma_f, withWeighting=True):
    nrOfPoints = int(J.shape[0]/2)

    if not withWeighting:
        sigma_f = sigma_u = sigma_v = 1

    weights = np.zeros([X_query.shape[1]*2,X_query.shape[1]*2])
    for i in range(X_query.shape[1]):

        u_hat_std = np.sqrt(sigma_u**2+((X_query[0,i])**2)/((X_query[2,i])**2)*sigma_f**2)
        v_hat_std = np.sqrt(sigma_v**2+((X_query[1,i])**2)/((X_query[2,i])**2)*sigma_f**2)
        weights[2*i,2*i] = 1/u_hat_std
        weights[2*i+1,2*i+1] = 1/v_hat_std

    Epsilon_p = np.linalg.inv(J.T @ weights @ J)

    # rVec_std = np.rad2deg(np.sqrt(np.diag(Epsilon_p))[:3])
    rVec_std = np.rad2deg(np.sqrt(np.diag(Epsilon_p))[:3])
    tVec_std = np.sqrt(np.diag(Epsilon_p))[3:]*1000
    
    printBool = True
    if printBool:
        print('\n\nStandard deviations - Task 4.1')
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
        print()


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


def estimateCovariance(imageName, nrOfSamples, model, runOwnImages=True, dataPath="storedData/", printBool=True):
    if runOwnImages:
        queryImage = f"model_images/{imageName}.jpg"
    else:
        queryImage = f"../hw5_data_ext/{imageName}.JPG"
    queryImg = cv.imread(queryImage)

    X = np.load(dataPath+f"/model/X.npy")
    K_ = np.loadtxt(f"calibration/results/calibrated_K.txt")
    modelDesc = np.load(dataPath+f"/model/descriptor.npy")
    matches = np.loadtxt(dataPath+f"/query/{imageName}_matches.txt").astype(int)
    u = np.loadtxt(dataPath+f"/query/{imageName}_u.txt")
    inliers = np.loadtxt(dataPath+f"/query/{imageName}_inliers.txt")


    # Initiate SIFT detector
    # sift = cv.SIFT_create()
    # qkp, qDes = sift.detectAndCompute(queryImg,None)

    # # BFMatcher with default params and k best matches
    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(qDes, modelDesc, k=2)

    # # Apply ratio test (see D.Lowe paper and opencv docs)
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance: # 0.75
    #         good.append([m])
    
    
    # u = np.array([qkp[i].pt for i in range(len(qkp))]).T
    # matchesIndexes = np.array([])
    # for i in range(np.array(good).size):
    #     matchesIndexes = np.append(matchesIndexes, np.array([good[i][0].queryIdx, good[i][0].trainIdx]))
    # matchesIndexes = matchesIndexes.reshape([len(good),2]).astype(int)

    X_dehom = (X.T[:,:3][matches[:, 1]]).T

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

        _, rotVec, transVec, inliers = cv.solvePnPRansac(X_dehom.T, u[:, matches[:, 0]].T, K, None)
        
        # inliers = inliers[:,0]
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

        


def calculateQueryCovariance(imageName, model, KPath, printBool=True):
    J = localize(model, KPath, imageName, runOwnImages, saveParams=False, useWeighted=True)
    nrOfPoints = int(J.shape[0]/2)

    task = "3.3"

    if task == "3.2":
        #  Task 3.2
        sigma_u = 1
        sigma_v = 1
    elif task == "3.3":
        # Task 3.3
        sigma_u = 50#**2
        sigma_v = 0.1#**2
    W = calculateWeights(sigma_u, sigma_v, nrOfPoints)
    Epsilon_p = np.linalg.inv(J.T @ W @ J)

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
    useSavedModel = True

    useRootSIFT = True
    task41 = True

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
            m2.performMatching(useRootSIFT=useRootSIFT)
            m2.generateStructure()
            m2.saveModel()

        localize(m2, "calibration/results/calibrated_K.txt", localizeImg, runOwnImages=True, useRootSIFT=useRootSIFT, task41=task41)
        visualize_query_results(localizeImg, runOwnImages=True)

        calculateQueryCovariance(localizeImg, m2, "calibration/results/calibrated_K.txt")
        estimateCovariance(localizeImg, 500, m2, dataPath="storedData/", printBool=True)
        

    else:
        localizeImg = "IMG_8210"

        m = model("../hw5_data_ext/IMG_8207.JPG", "../hw5_data_ext/IMG_8229.JPG", "../hw5_data_ext/K.txt", 6.2)

        if useSavedModel:
            m.loadModel("storedData/model") 

        else:
            m.performMatching()
            m.generateStructure()
            m.saveModel()

        estimateCovariance(localizeImg, 500, m, dataPath="storedData/", printBool=True)

        localize(m, "../hw5_data_ext/K.txt", localizeImg, runOwnImages=False, useRootSIFT=useRootSIFT, task41=task41)
        
        calculateQueryCovariance(localizeImg, m, "../hw5_data_ext/K.txt")
        visualize_query_results(localizeImg, runOwnImages=False)



