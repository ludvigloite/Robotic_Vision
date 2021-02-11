import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

grassImg = mpimg.imread('./data/grass.jpg')
#print(grassImg.shape)
width = grassImg.shape[1]
height = grassImg.shape[0]
print("width:", grassImg.shape[1], "height:", grassImg.shape[0])
print("ny")

threshold = 100

newImg = grassImg[:,:,1] > threshold

#plt.imshow(newImg, cmap="gray")
#plt.show()

normalizedImg = grassImg/np.sum(grassImg, axis=-1)[:,:,None]

plt.imshow(normalizedImg[:,:,2])
plt.show()

threshold_norm = 0.4

plt.imshow(normalizedImg[:,:,1] > threshold_norm, cmap="gray")

#plt.show()


normalizedThresholdImg = normalizedImg[:,:,1] > threshold

plt.imshow(normalizedThresholdImg)

#plt.show()

plt.imshow(normalizedImg[:,:,0])
#plt.show()

plt.imshow(normalizedImg[:,:,1])
#plt.show()

plt.imshow(normalizedImg[:,:,2])
#plt.show()

#plt.imshow(newImg)
