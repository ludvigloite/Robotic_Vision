import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


boardSize = (7,10)
shape = (4032, 3024)

run_calibration = True
save_plots = True
image_to_undistort = "IMG_3926"


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('calibration/calibration_jpg/*.jpg')

if run_calibration:
    print("Running calibration")
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        #cv.imwrite('calibration/results/numero1.png', gray)
        ret, corners = cv.findChessboardCorners(gray, boardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("found chessboard!")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, boardSize, corners2, ret)
            #cv.imshow('img', img)
            cv.waitKey(10)
        else:
            print("did not find chessboard..")
            
    cv.destroyAllWindows()

    np.save('storedData/calibration/imgpoints.npy', imgpoints)
    np.save('storedData/calibration/objpoints.npy', objpoints)

else:
    ## Save objpoints and imgpoints ##
    imgpoints = np.load('storedData/calibration/imgpoints.npy',allow_pickle=True)
    objpoints = np.load('storedData/calibration/objpoints.npy',allow_pickle=True)


###### Calibration ######

print("Calibrating")

ret, mtx, dist, rvecs, tvecs, std_int,_,_ = cv.calibrateCameraExtended(objpoints, imgpoints, shape, None, None)

###### Reprojection error ######

image_numbers = []
errors = []

fig = plt.figure(figsize = (10,5))

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    error_2d = imgpoints2-imgpoints[i]
    plt.scatter(error_2d[:,0][:,0],error_2d[:,0][:,1])
    errors.append(error)
    image_numbers.append(i+1)
    mean_error += error

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Scatter plot of 2D error vectors for all {len(objpoints)} images')
if save_plots:
    plt.savefig('calibration/results/2d_scatter.png')
plt.clf()

print("\nTotal error: {}\n".format(mean_error/len(objpoints)) )

# Ordering of elements taken from cv documentation

# Printing 

print('Standard errors')
print('\nFocal length and principal point')
print('--------------------------------')
print('fx: %g +/- %g' % (mtx[0,0], std_int[0]))
print('fy: %g +/- %g' % (mtx[1,1], std_int[1]))
print('cx: %g +/- %g' % (mtx[0,2], std_int[2]))
print('cy: %g +/- %g' % (mtx[1,2], std_int[3]))
print('\nDistortion coefficients')
print('--------------------------------')
print('k1: %g +/- %g' % (dist[0,0], std_int[4]))
print('k2: %g +/- %g' % (dist[0,1], std_int[5]))
print('p1: %g +/- %g' % (dist[0,2], std_int[6]))
print('p2: %g +/- %g' % (dist[0,3], std_int[7]))
print('k3: %g +/- %g' % (dist[0,4], std_int[8]))

params = np.array([dist[0,0], dist[0,1]]).ravel()
np.savetxt("calibration/results/dist_params.txt", params)

plt.bar(image_numbers,errors)
plt.xlabel('Image Number')
plt.ylabel('Mean reprojection error [pixels]')
plt.title('Mean reprojection error per image')
if save_plots:
    plt.savefig('calibration/results/Reprojection_errors.png')
plt.clf()


###### Undistortion ######

#imageToUndistort = f'calibration/calibration_jpg/{image_to_undistort}.jpg'
image_to_undistort = 'IMG_3895'
imageToUndistort = f'model_images/{image_to_undistort}.jpg'

img = cv.imread(imageToUndistort)
h,  w = img.shape[:2]
mtx_original = mtx
dist_original = dist

fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .05, wspace=.3)
axs = axs.ravel()

newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
np.savetxt("calibration/results/calibrated_K.txt", newcameramtx)

# undistort single image
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite(f'calibration/results/calib_result_{image_to_undistort}.jpg', dst)
print(f'Saved {image_to_undistort}')




"""
images_to_undistort = ['IMG_3895','IMG_3896','IMG_3897','IMG_3898','IMG_3899','IMG_3900','IMG_3901']
for image_to_undistort in images_to_undistort:
    imageToUndistort = f'model_images/{image_to_undistort}.jpg'

    img = cv.imread(imageToUndistort)
    h,  w = img.shape[:2]
    mtx_original = mtx
    dist_original = dist

    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .05, wspace=.3)
    axs = axs.ravel()

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    np.savetxt("calibration/results/calibrated_K.txt", newcameramtx)

    # undistort single image
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f'model_images/calibrated/{image_to_undistort}.jpg', dst)
    print(f'Saved {image_to_undistort}')
"""

for i in range(10):
    mtx[0,0] = np.random.normal(mtx_original[0,0],std_int[0])
    mtx[1,1] = np.random.normal(mtx_original[1,1],std_int[1])
    mtx[0,2] = np.random.normal(mtx_original[0,2],std_int[2])
    mtx[1,2] = np.random.normal(mtx_original[1,2],std_int[3])

    dist[0,0] = np.random.normal(dist_original[0,0],std_int[4])
    dist[0,1] = np.random.normal(dist_original[0,1],std_int[5])
    dist[0,2] = np.random.normal(dist_original[0,2],std_int[6])
    dist[0,3] = np.random.normal(dist_original[0,3],std_int[7])
    dist[0,4] = np.random.normal(dist_original[0,4],std_int[8])


    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst_cropped = dst[y:y+h, x:x+w]

    axs[i].imshow(dst_cropped)
    axs[i].set_title(f"distortion #{i+1}")

if save_plots:
    plt.savefig('calibration/results/Comparison.png')
plt.clf()












"""
def calibration(run_calibration=True):
    boardSize = (7,10)
    shape = (4032, 3024)
    image_to_undistort = "IMG_3915"

    #Må ha samme focus når vi tar bilde av bygning og checkerboard.



    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('calibration/calibration_jpg/*.jpg')

    if run_calibration:
        print("This might take some time, because the images are not taken in dark environment, and with other elements also in images")
        for fname in images:
            print(fname)
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            #cv.imwrite('calibration/results/numero1.png', gray)
            ret, corners = cv.findChessboardCorners(gray, boardSize, None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print("found chessboard!")
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, boardSize, corners2, ret)
                #cv.imshow('img', img)
                cv.waitKey(10)
            else:
                print("did not find chessboard..")
                
        cv.destroyAllWindows()

        np.save('storedData/calibration/imgpoints.npy', imgpoints)
        np.save('storedData/calibration/objpoints.npy', objpoints)

    else:
        ## Save objpoints and imgpoints ##
        imgpoints = np.load('storedData/calibration/imgpoints.npy',allow_pickle=True)
        objpoints = np.load('storedData/calibration/objpoints.npy',allow_pickle=True)



    ###### Calibration ######

    ret, mtx, dist, rvecs, tvecs, std_int,_,_ = cv.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs, std_int

def getReprojectionError(mtx, dist, rvecs, tvecs, std_int,save_plots = True):
    ###### Reprojection error ######

    imgpoints = np.load('storedData/calibration/imgpoints.npy',allow_pickle=True)
    objpoints = np.load('storedData/calibration/objpoints.npy',allow_pickle=True)

    image_numbers = []
    errors = []

    fig = plt.figure(figsize = (10,5))

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        error_2d = imgpoints2-imgpoints[i]
        plt.scatter(error_2d[:,0][:,0],error_2d[:,0][:,1], marker='x')
        errors.append(error)
        image_numbers.append(i+1)
        mean_error += error

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Scatter plot of 2D error vectors for all {len(objpoints)} images')
    if save_plots:
        plt.savefig('calibration/results/2d_scatter.png')
    plt.clf()

    print("\nTotal error: {}\n".format(mean_error/len(objpoints)) )

    # Ordering of elements taken from cv documentation

    print('Standard errors')
    print('\nFocal length and principal point')
    print('--------------------------------')
    print('fx: %g +/- %g' % (mtx[0,0], std_int[0]))
    print('fy: %g +/- %g' % (mtx[1,1], std_int[1]))
    print('cx: %g +/- %g' % (mtx[0,2], std_int[2]))
    print('cy: %g +/- %g' % (mtx[1,2], std_int[3]))
    print('\nDistortion coefficients')
    print('--------------------------------')
    print('k1: %g +/- %g' % (dist[0,0], std_int[4]))
    print('k2: %g +/- %g' % (dist[0,1], std_int[5]))
    print('p1: %g +/- %g' % (dist[0,2], std_int[6]))
    print('p2: %g +/- %g' % (dist[0,3], std_int[7]))
    print('k3: %g +/- %g' % (dist[0,4], std_int[8]))

    params = np.array([dist[0,0], dist[0,1]]).ravel()
    np.savetxt("calibration/results/dist_params.txt", params)

    plt.bar(image_numbers,errors)
    plt.xlabel('Image Number')
    plt.ylabel('Mean reprojection error [pixels]')
    plt.title('Mean reprojection error per image')
    if save_plots:
        plt.savefig('calibration/results/Reprojection_errors.png')
    plt.clf()


def distortMany():

    ###### Undistortion ######

    imageToUndistort = f'calibration/calibration_jpg/{image_to_undistort}.jpg'

    img = cv.imread(imageToUndistort)
    h,  w = img.shape[:2]
    mtx_original = mtx
    dist_original = dist

    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .05, wspace=.3)
    axs = axs.ravel()

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    np.savetxt("calibration/results/calibrated_K.txt", newcameramtx)

    for i in range(10):
        mtx[0,0] = np.random.normal(mtx_original[0,0],std_int[0])
        mtx[1,1] = np.random.normal(mtx_original[1,1],std_int[1])
        mtx[0,2] = np.random.normal(mtx_original[0,2],std_int[2])
        mtx[1,2] = np.random.normal(mtx_original[1,2],std_int[3])

        dist[0,0] = np.random.normal(dist_original[0,0],std_int[4])
        dist[0,1] = np.random.normal(dist_original[0,1],std_int[5])
        dist[0,2] = np.random.normal(dist_original[0,2],std_int[6])
        dist[0,3] = np.random.normal(dist_original[0,3],std_int[7])
        dist[0,4] = np.random.normal(dist_original[0,4],std_int[8])


        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        #cv.imwrite('calibration/results/calib_result_3612.png', dst)

        axs[i].imshow(dst)
        axs[i].set_title(f"distortion #{i+1}")

    if save_plots:
        plt.savefig('calibration/results/Comparison.png')
    plt.clf()

def undistortImage(image_to_undistort, mtx, dist, saveFig = True, crop=False):
    imageToUndistort = f'calibration/calibration_jpg/{image_to_undistort}.jpg'

    img = cv.imread(imageToUndistort)
    h,  w = img.shape[:2]

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    np.savetxt("calibration/results/calibrated_K.txt", newcameramtx)

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    if crop:
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
    if saveFig:
        cv.imwrite(f'calibration/results/calib_result{image_to_undistort}', dst)


    
img = img[y:y+h, x:x+w]

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(dst)
ax2 = fig.add_subplot(1,2,2)
ax1.imshow(img)
if save_plots:
    plt.savefig('calibration/results/Comparison.png')




if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs, std_int = calibration()
    getReprojectionError(mtx, dist, rvecs, tvecs, std_int)

    image_to_undistort = "IMG_3915"
    undistortImage(image_to_undistort, mtx, dist)


"""