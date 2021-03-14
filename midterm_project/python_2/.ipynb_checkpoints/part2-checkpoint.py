import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from methods import *
#from quanser import Quanser
#from generate_quanser_summary import *
from common import *

K = np.loadtxt('../data/K.txt')
platform_corners_metric = np.loadtxt('../data/platform_corners_metric.txt')
platform_corners_image = np.loadtxt('../data/platform_corners_image.txt')


# Task 2.1
uv1 = np.vstack((platform_corners_image, np.ones(platform_corners_image.shape[1])))
xy = np.linalg.inv(K) @ uv1

H = estimate_H(xy, platform_corners_metric)

XY = np.delete(platform_corners_metric, 2, 0)
uv_from_H = K @ H @ XY
uv_from_H = uv_from_H[:2] / uv_from_H[2]

T = decompose_H(H) 
Rt = T[:3,:]

uv_from_Rt = K @ Rt @ platform_corners_metric
uv_from_Rt = uv_from_Rt[:2] / uv_from_Rt[2]

T_true = np.loadtxt('../data/platform_to_camera.txt')

img = mpimg.imread('../data/video0000.jpg')
plt.imshow(img)
plt.xlim([200, 450])
plt.ylim([550, 425])
plt.scatter(uv_from_H[0], uv_from_H[1], color='red')
plt.scatter(uv_from_Rt[0], uv_from_Rt[1])
plt.legend(['Coordinated calculated using KH','Coordinated calculated using [R t]'])
plt.savefig('plot_task21')
plt.show()
print(f'Reprojection error using H: {np.round(list(map(lambda l: np.linalg.norm(l), (platform_corners_image-uv_from_H).T)),3)}')
print(f'Reprojection error using Rt: {np.round(list(map(lambda l: np.linalg.norm(l), (platform_corners_image-uv_from_Rt).T)),3)}')


# Task 2.2
R0 = np.zeros((4,4))
R0[:3,:3] = T[:3,:3]
R0[3,3] = 1
p1 = 0
p2 = 0
p3 = 0
t = T[:3,3]

# For task 2.3
task23 = False
if task23:
    platform_corners_metric = platform_corners_metric[:,:3]
    platform_corners_image = platform_corners_image[:,:3]
    R0[:3,:3] = -R0[:3,:3]
    t = -t

p = np.array([p1, p2, p3])
p = np.concatenate((p, t))

R = rotate_x(p[0]) @ rotate_x(p[1]) @ rotate_x(p[2]) @ R0
Rt = np.hstack((R[:3,:3], np.reshape(p[3:], (3,1))))
uv_from_Rt = K @ Rt @ platform_corners_metric
uv_hat = uv_from_Rt[:2] / uv_from_Rt[2]

print("Correct values: ", platform_corners_image)
print("Initial guess: ", uv_hat)
print(Rt)

def residuals2(uv, K, R0, platform_corners_metric, p):
    R = rotate_x(p[0]) @ rotate_y(p[1]) @ rotate_z(p[2]) @ R0
    Rt = np.hstack((R[:3,:3], np.reshape(p[3:], (3,1))))
    uv_from_Rt = K @ Rt @ platform_corners_metric
    uv_hat = uv_from_Rt[:2] / uv_from_Rt[2]

    n = uv.shape[1]
    r = np.zeros(2*n)

    for i in range(n):
        r[i] = uv_hat[0][i] - uv[0][i]
        r[n+i] = uv_hat[1][i] - uv[1][i]
    
    return r


rotationfun = lambda p: residuals2(platform_corners_image, K, R0, platform_corners_metric, p)
p = levenberg_marquardt(rotationfun, p)
r = rotationfun(p)

R = rotate_x(p[0]) @ rotate_y(p[1]) @ rotate_z(p[2]) @ R0
Rt = np.hstack((R[:3,:3], np.reshape(p[3:], (3,1))))
uv_from_Rt = K @ Rt @ platform_corners_metric
uv_final = uv_from_Rt[:2] / uv_from_Rt[2]

#print("Final guess: ", uv_final)

print(f'Reprojection error task 2.2: {list(map(lambda l: np.linalg.norm(l), (platform_corners_image-uv_final).T))}')

"""img = mpimg.imread('../data/video0000.jpg')
plt.imshow(img)
plt.xlim([200, 450])
plt.ylim([550, 425])
plt.scatter(platform_corners_image[0], platform_corners_image[1], color='red', s=100)
plt.scatter(uv_final[0], uv_final[1], s=20)
plt.scatter(uv_hat[0], uv_hat[1], s=10, color='green')
plt.legend(['Ground truth', 'Final estimate', 'Initial estimate'])
plt.show()"""