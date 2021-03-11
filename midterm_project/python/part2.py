import matplotlib.pyplot as plt
import numpy as np
from common import *
from quanser import Quanser
from methods import *
from generate_quanser_summary import *

plat_corners_img = np.loadtxt('data/platform_corners_image.txt')
XY = np.loadtxt('data/platform_corners_metric.txt')
K = np.loadtxt('data/K.txt')

quanser = Quanser()

fig = plt.figure(figsize=plt.figaspect(0.35))

uv = plat_corners_img

n = uv.shape[1]


uv1 = np.vstack((uv, np.ones(n)))
XY2 = XY[:2,:]
xy = (np.linalg.inv(K)@uv1)[:2]      
H = estimate_H(xy, XY2)

T1,T2 = decompose_H(H)

T = T1

x_tilde = H @ np.vstack((XY2, np.ones(n)))
uv_from_H_a = K@x_tilde
#print("uv:_" ,uv_from_H_a)

x_b = np.vstack((XY2, np.zeros(n), np.ones(n)))

H_b = T[:3,:]
#print("H_b ",H_b)
uv_b = K @ H_b @ x_b

task21 = False
task22 = True

if task21:

    imgName = '2.1a'
    imgNr = 1
    uv_from_H = uv_from_H_a

    which_transform = 'b'

    if which_transform == 'b':
        imgName = '2.1b'
        imgNr = 2
        uv_from_H = uv_b

    uv_from_H = uv_from_H[:2]/uv_from_H[-1]

    diff = np.linalg.norm(uv - uv_from_H, axis=0)
    diff_max = np.amax(diff)
    diff_min = np.amin(diff)
    diff_mean = np.mean(diff)
    print(
        f'\nPlatform Image'
        f'\nDiffs: {diff}'
        f' \naverage: {diff_mean} \nmin: {diff_min} \nmax: {diff_max}\n')


    plt.clf()
    generate_figure(fig, imgNr, K, T, uv, uv_from_H, XY)
    #plt.savefig('data/out{}.png'.format(imgName))
    #plt.show()

elif task22:

    R0 = T[:3,:3]
    print("R0: ", R0)
    t1,t2,t3 = T[:3,3]

    p = [0,0,0,t1,t2,t3]
    
    residualsfun = lambda p : quanser.residuals2(uv, p, XY2, K, R0)

    p = levenberg_marquardt2(residualsfun,p)
    
    R = rotate_x(p[0])[:3,:3] @ rotate_y(p[1])[:3,:3] @ rotate_z(p[2])[:3,:3] @ R0
    t = np.vstack([p[3],p[4],p[5]])
    T_b = np.hstack([R,t])
    T = np.vstack((T_b,[0,0,0,1]))

    x_b = np.vstack((XY2, np.zeros(n), np.ones(n)))
    uv_from_T = K @ T_b @ x_b
    uv_from_T = uv_from_T[:2]/uv_from_T[-1]


    imgNr = 4
    imgName = '2.2'

    
    plt.clf()
    generate_figure(fig, imgNr, K, T, uv, uv_from_T, XY)
    plt.savefig('data/out{}.png'.format(imgName))
    #plt.show()

    # Note:
    # The generated figures will be saved in your working
    # directory under the filenames out_*.png.

    #generate_quanser_summary(trajectory, all_residuals, detections)
    #r = rotate_x(p_1) @ rotate_y(p_2) @ rotate_z(p_3) @ R0
