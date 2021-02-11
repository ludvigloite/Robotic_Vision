import numpy as np

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    return (R + G + B)/3

def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """

    kernel = np.array([0.5,0,-0.5])

    Ix = np.array([np.convolve(I[i,:], kernel, mode="same") for i in range(I.shape[0])])
    Iy = np.array([np.convolve(I[:,i], kernel, mode="same") for i in range(I.shape[1])]).T
    Im = np.sqrt(Ix**2 + Iy**2)

   
    

    return Ix, Iy, Im

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations:
    # h = 2*np.ceil(3*sigma) + 1.

    h = int(np.ceil(3*sigma))
    discrete_kernel = np.array([1/(2*np.pi*sigma**2)*np.exp(-i**2/(2*sigma**2)) for i in range(-h, h)])
    temp_result = np.array([np.convolve(I[i,:], discrete_kernel, mode="same") for i in range(I.shape[0])]) 
    result = np.array([np.convolve(temp_result[:,i], discrete_kernel, mode="same") for i in range(temp_result.shape[1])]).T
    

    return result

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    above_thr = np.array([])
    rows, cols = Im.shape
    for row in range(rows):
        for col in range(cols):
            if Im[row, col] > threshold:
                theta = np.arctan2(Iy[row, col], Ix[row, col])
                above_thr = np.append(above_thr, np.array([int(col), int(row), theta]))

    return above_thr.reshape((-1, 3))[:,0], above_thr.reshape((-1, 3))[:,1], above_thr.reshape((-1, 3))[:,2]
  
