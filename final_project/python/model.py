import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy

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

    def performMatching(self, useRootSIFT=False):
        # Initiate SIFT detector
        sift = cv.SIFT_create(100000)
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1,None) # query image
        kp2, des2 = sift.detectAndCompute(self.img2,None) # train image

        if useRootSIFT:
            des1 /= des1.sum(axis=1, keepdims=True)
            des1 = np.sqrt(des1)
            des2 /= des2.sum(axis=1, keepdims=True)
            des2 = np.sqrt(des2)
            #finner flere matcher. Får inliers. fjerner verste punktene

        print(f"descriptor 1 shape: {des1.shape}")
        print(f"descriptor 2 shape: {des2.shape}")

        # BFMatcher with default params and k best matches
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test (see D.Lowe paper and opencv docs)
        good = []
        ratio = 0.75
        for m,n in matches:
            if m.distance < ratio*n.distance: # 0.75
                good.append([m])

        print(f"Number of good matches: {len(good)}")

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


        
        
    def generateStructure(self,fromSavedModel=False):
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
        draw_point_cloud(self.X, self.img1, uv1, xlim=[-5,+5], ylim=[-5,+5], zlim=[2,15])

        #plt.show()

    def getDescriptor(self):
        return self.descriptor

    def getObjectPoints(self):
        return self.X

        #implement 4.4 python legge til 2 linjer

    def saveModel(self):
        np.save("storedData/model/descriptor.npy", self.descriptor)
        np.save("storedData/model/X.npy", self.X)
        np.savetxt("storedData/model/K.txt", self.K)

        np.savetxt("storedData/model/allMatches_img1.txt", self.allMatches["img1"])
        np.savetxt("storedData/model/allMatches_img2.txt", self.allMatches["img2"])
        
        print("Model saved")
    
    def loadModel(self, dataPath):
        self.descriptor = np.load(dataPath+"/descriptor.npy")
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
    m = model("../hw5_data_ext/IMG_8207.JPG", "../hw5_data_ext/IMG_8229.JPG", "../hw5_data_ext/K.txt", 6.2)

    #m2 = model("model_images/IMG_3929.jpg", "model_images/IMG_3930.jpg", "calibration/results/calibrated_K.txt")
    #m2 = model("model_images/IMG_3898.jpg", "model_images/IMG_3899.jpg", "calibration/results/calibrated_K.txt")

    sparse = createSparsityMatrix(10)


    useSavedModel = False

    if useSavedModel:
        m2.loadModel("storedData/model") 
        m2.generateStructure(fromSavedModel=True)

    else:
        m.performMatching()
        m.generateStructure()
        m.saveModel()
    
    #m.generateStructure()    