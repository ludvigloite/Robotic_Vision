import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
import pydegensac
from time import time
from copy import deepcopy

from hw5.figures import *
from hw5.estimate_E import *
from hw5.decompose_E import *
from hw5.triangulate_many import *
from hw5.epipolar_distance import *
from hw5.estimate_E_ransac import *
from hw5.F_from_E import *
from LM import *
from util import *
from localize import * 

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def verify_cv2_fundam(kps1, kps2, tentatives, th = 1.0 , n_iter = 10000):
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentatives ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentatives ]).reshape(-1,2)
    F, mask = cv.findFundamentalMat(src_pts, dst_pts, cv.RANSAC, th, 0.999, n_iter)
    print ('cv2 found {} inliers'.format(int(deepcopy(mask).astype(np.float32).sum())))
    return F, mask

def verify_pydegensac_fundam(kps1, kps2, tentatives, th = 1.0,  n_iter = 10000):
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentatives ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentatives ]).reshape(-1,2)
    F, mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, th, 0.999, n_iter, enable_degeneracy_check= True)
    print ('pydegensac found {} inliers'.format(int(deepcopy(mask).astype(np.float32).sum())))
    return F, mask

def draw_matches(kps1, kps2, tentatives, img1, img2, H, mask):
    if H is None:
        print ("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h,w,ch = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts, H)
    #print (dst)
    #Ground truth transformation
    dst_GT = cv.perspectiveTransform(pts, H_gt)
    img2_tr = cv.polylines(decolorize(img2),[np.int32(dst)],True,(0,0,255),3, cv.LINE_AA)
    img2_tr = cv.polylines(deepcopy(img2_tr),[np.int32(dst_GT)],True,(0,255,0),3, cv.LINE_AA)
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img_out = cv.drawMatches(decolorize(img1),kps1,img2_tr,kps2,tentatives,None,**draw_params)
    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img_out, interpolation='nearest')
    plt.show()

def decolorize(img):
    return  cv.cvtColor(cv.cvtColor(img,cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)

def draw_matches_F(kps1, kps2, tentatives, img1, img2, mask, name):
    matchesMask = mask.ravel().tolist()
    h,w,ch = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img_out = cv.drawMatches(decolorize(img1),kps1,
                              decolorize(img2),kps2,tentatives,None,**draw_params)
    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img_out, interpolation='nearest')
    plt.savefig(f'results/4.5_{name}.png')
    

class model:
    def __init__(self, img1Path, img2Path, KPath, scalingFactor):
        self.img1 = cv.imread(img1Path)
        self.img2 = cv.imread(img2Path)
        self.K = np.loadtxt(KPath)
        self.X = None # 3D image points
        self.descriptor = None # Descriptor
        self.scaling_factor = scalingFactor #6.2 #Given images

    def meanReprojectionError(self, uv1, uv2):
        X2 = self.T @ self.X

        uv1_proj = project(self.K,self.X)
        uv2_proj = project(self.K,X2)

        r1 = np.mean(np.sqrt((uv1[0,:] - uv1_proj[0,:])**2 + (uv1[1,:] - uv1_proj[1,:]) ** 2))
        r2 = np.mean(np.sqrt((uv2[0,:] - uv2_proj[0,:])**2 + (uv2[1,:] - uv2_proj[1,:]) ** 2))

        return r1, r2

    def performMatching(self):
        # Initiate SIFT detector
        sift = cv.SIFT_create(100000)
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1,None) # query image
        kp2, des2 = sift.detectAndCompute(self.img2,None) # train image

        for i in range(len(kp1)):
            kp1[i].size = 5*kp1[i].size
        for i in range(len(kp2)):
            kp2[i].size = 5*kp2[i].size

        print(f"descriptor 1 shape: {des1.shape}")
        print(f"descriptor 2 shape: {des2.shape}")

        # BFMatcher with default params and k best matches
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        good = []
        good2 = []
        ratio = 0.75
        for m,n in matches:
            if m.distance < ratio*n.distance: # 0.75
                good.append([m])
                good2.append(m)

        print(f"Number of good matches: {len(good)}")

        th = 0.5
        n_iter = 50000

        t=time()
        cv_H, cv_mask = verify_cv2_fundam(kp1,kp2,good2, th, n_iter )
        print ("{0:.5f}".format(time()-t), ' sec cv2')

        t=time()
        cmp_H, cmp_mask = verify_pydegensac_fundam(kp1,kp2,good2, th, n_iter)
        print ("{0:.5f}".format(time()-t), ' sec pydegensac')

        #draw_matches_F(kp1, kp2, good2, self.img1, self.img2, cv_mask, "cv")
        #draw_matches_F(kp1, kp2, good2, self.img1, self.img2, cmp_mask,"pydegensac")

        self.degensac_mask = cmp_mask
        self.cv_mask = cv_mask


        # Obtain image indeces
        self.allMatches = []
        allPts1 = []
        allPts2 = []
        desc1Idx = np.zeros(des1.shape[0]).astype(bool)
        desc2Idx = np.zeros(des2.shape[0]).astype(bool)
        for i in range(np.array(good).size):
            desc1Idx[good[i][0].queryIdx] = True
            desc2Idx[good[i][0].trainIdx] = True
            pt1 = np.array(kp1[good[i][0].queryIdx].pt)
            pt2 = np.array(kp2[good[i][0].trainIdx].pt)
            allPts1 = np.append(allPts1, pt1)
            allPts2 = np.append(allPts2, pt2)

        allPts1 = np.reshape(allPts1,[int(allPts1.size/2),2])
        allPts2 = np.reshape(allPts2,[int(allPts2.size/2),2])
        self.allMatches = { "img1" : allPts1,
                            "img2" : allPts2}
                            
        # self.allDes = {"des1" : des1, "des2" : des2}
        self.descriptor = des1[desc1Idx]
        self.kp1 = kp1


        
        
    def generateStructure(self, whichRansac, fromSavedModel=False):
        '''
        Generate 3D points from the image set
        '''
        n = self.allMatches["img1"].shape[0]
        uv1 = np.vstack([self.allMatches["img1"].T, np.ones(n)])
        uv2 = np.vstack([self.allMatches["img2"].T, np.ones(n)])
        xy1 = np.linalg.inv(self.K)@uv1
        xy2 = np.linalg.inv(self.K)@uv2

        confidence = 0.99
        inlier_fraction = 0.50
        distance_threshold = 4.0
        num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)
        print('Running RANSAC: %d trials, %g pixel threshold' % (num_trials, distance_threshold))
        E,inliers = estimate_E_ransac(xy1, xy2, self.K, distance_threshold, num_trials)
        
        print("Nu of inliers Standard Ransac: ",np.count_nonzero(inliers))
        print("Nu of inliers Degensac: ",np.count_nonzero(self.degensac_mask))
        print("Nu of inliers cv: ",np.count_nonzero(self.cv_mask))

        if whichRansac == "degensac":
            inliers = self.degensac_mask
        elif whichRansac == "cv":
            inliers = self.cv_mask.T[0]


        uv1 = uv1[:,inliers]
        uv2 = uv2[:,inliers]
        xy1 = xy1[:,inliers]
        xy2 = xy2[:,inliers]

        
        if not fromSavedModel:
            self.descriptor = self.descriptor[inliers]

        E = estimate_E(xy1, xy2)
        
        # Find the correct pose
        T4 = decompose_E(E)
        best_num_visible = 0
        for i, T in enumerate(T4):
            P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
            P2 = T[:3,:]
            X1 = triangulate_many(xy1, xy2, P1, P2)
            X2 = T@X1
            num_visible = np.sum((X1[2,:] > 0) & (X2[2,:] > 0))
            if num_visible > best_num_visible:
                best_num_visible = num_visible
                best_T = T
                best_X1 = X1
        self.T = best_T
        self.X = best_X1
        # print('Best solution: %d/%d points visible' % (best_num_visible, xy1.shape[1]))

        ########### Reprojection before bundle adjustment ###########

        r1,r2 = self.meanReprojectionError(uv1, uv2)
        print(f"Before Bundle Adjustment:\nMean reprojection error train_img: {r1}\nMean reprojection error query_img: {r2}")

        ########### Bundle adjustment ###########
        nrOf3DPoints = self.X.shape[1]
        params0 = np.zeros(6+3*nrOf3DPoints)
        sparsity_matrix = createSparsityMatrix(nrOf3DPoints)

        LS_result = scipy.optimize.least_squares(fun=residuals2,x0=params0, jac_sparsity=sparsity_matrix, verbose=1, args=(self.X,self.T, uv1, uv2, self.K, nrOf3DPoints))
        
        params = LS_result.x
        self.X = self.X + np.vstack((params[6:].reshape((nrOf3DPoints,3)).T, np.zeros(nrOf3DPoints)))
        T_query = compose_T2(params[:6], self.T)

        ########### Reprojection after bundle adjustment ###########

        X_query = T_query @ self.X

        uv1_proj = project(self.K,self.X)
        uv2_proj = project(self.K,X_query)

        r1 = np.mean(np.sqrt((uv1[0,:] - uv1_proj[0,:])**2 + (uv1[1,:]- uv1_proj[1,:]) ** 2))
        r2 = np.mean(np.sqrt((uv2[0,:] - uv2_proj[0,:])**2 + (uv2[1,:]- uv2_proj[1,:]) ** 2))

        print(f"After Bundle Adjustment:\nMean reprojection error train_img: {r1}\nMean reprojection error query_img: {r2}")

        ########### Saving and printing ###########

        self.allMatches["3D-points"] = self.X
        X1_opt = self.X
        self.X[:3,:] = self.X[:3,:] * self.scaling_factor
        self.allMatches["3D-points"] = self.X

        uv2_opt = to_homogeneous(uv2_proj)

        # np.random.seed(123) # Comment out to get a random selection each time
        draw_correspondences(self.img1, self.img2, uv1, uv2_opt, F_from_E(E, self.K), sample_size=8)
        #draw_point_cloud(self.X, self.img1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])
        #draw_point_cloud(self.X, self.img1, uv1, xlim=[-10,+10], ylim=[-10,+10], zlim=[1,30])
        draw_point_cloud(self.X, self.img1, uv1, xlim=[-7.5,+7.5], ylim=[-7.5,+7.5], zlim=[1,10])

        #draw_point_cloud(self.X, self.img1, uv1, xlim=[-5,+5], ylim=[-5,+5], zlim=[1,5])

        plt.show()


    def saveModel(self):
        np.save("storedData/model/descriptor.npy", self.descriptor)
        #np.save("storedData/model/kp1.npy", self.kp1)
        np.save("storedData/model/X.npy", self.X)
        np.savetxt("storedData/model/K.txt", self.K)

        np.savetxt("storedData/model/allMatches_img1.txt", self.allMatches["img1"])
        np.savetxt("storedData/model/allMatches_img2.txt", self.allMatches["img2"])
        
        print("Model saved")
    
    def loadModel(self, dataPath):
        self.descriptor = np.load(dataPath+"/descriptor.npy")
        #self.kp1 = np.load(dataPath+"/kp1.npy")
        self.X = np.load(dataPath+"/X.npy")
        self.K = np.loadtxt(dataPath+"/K.txt")

        self.allMatches = []
        allPts1 = np.loadtxt(dataPath+"/allMatches_img1.txt")
        allPts2 = np.loadtxt(dataPath+"/allMatches_img2.txt")

        self.allMatches = { "img1" : allPts1,
                            "img2" : allPts2}

        print("Model loaded")

#4.1 , 4.2 halvveis, 4.3 ok, 4.4 helt good.
def createSparsityMatrix(nuWorldPoints, show=False):
    m = nuWorldPoints*2*2
    n = 6 + nuWorldPoints*3
    matrix = lil_matrix((m, n), dtype=int)
    for i in range(nuWorldPoints*2):
        for j in range(6):
            matrix[i,j] = 1

    for i in range(nuWorldPoints):
        for j in range(3):
            matrix[i*2,i*3+6+j] = 1
            matrix[i*2+1,i*3+6+j] = 1
            matrix[i*2+nuWorldPoints*2,i*3+6+j] = 1
            matrix[i*2+1+nuWorldPoints*2,i*3+6+j] = 1

    if show:
        plt.imshow(matrix.toarray())
        plt.savefig(f'results/sparsityMatrix_{nuWorldPoints}.png')
        plt.show()

    return matrix


if __name__ == "__main__":
    
    runOwnImages = True
    whichRansac = "degensac"
    useSavedModel = False 

    if runOwnImages:
        m2 = model("model_images/IMG_3898.jpg", "model_images/IMG_3899.jpg", "calibration/results/calibrated_K.txt",2.05)

        if useSavedModel:
            m2.loadModel("storedData/model") 

        else:
            m2.performMatching()
            m2.generateStructure(whichRansac=whichRansac)
            m2.saveModel()
        

    else:
        m = model("../hw5_data_ext/IMG_8207.JPG", "../hw5_data_ext/IMG_8229.JPG", "../hw5_data_ext/K.txt", 6.2)

        if useSavedModel:
            m.loadModel("storedData/model") 

        else:
            m.performMatching()
            m.generateStructure(whichRansac=whichRansac)
            m.saveModel()  