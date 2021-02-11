import numpy as np
import matplotlib.pyplot as plt
from common import *


K = np.loadtxt('../data/heli_K.txt')
T_plat2cam = np.loadtxt('../data/platform_to_camera.txt')
heli_points = np.loadtxt('../data/heli_points.txt')

filename = "../data/quanser.jpg"

quanserImg = plt.imread(filename)

scale = 0.05

plt.figure()
plt.imshow(quanserImg)

draw_frame(K,T_plat2cam,scale)

d = .1145
screws = np.array([[0,0,0,1],[d,0,0,1],[d,d,0,1],[0,d,0,1]]).T

u,v = project(K,T_plat2cam @ screws)
plt.scatter(u, v, c='y', marker='.', s=40)


psi = 11.6
theta = 28.9
phi = 0

T_plat2base = translate_x(d/2) @ translate_y(d/2) @ rotate_z(psi)
T_base2hinge = translate_z(0.325) @ rotate_y(theta)
T_hinge2arm = translate_z(-0.05)
T_arm2rotors = translate_x(0.65) @ translate_z(-0.03) @ rotate_x(phi)

draw_frame(K,T_plat2cam @ T_plat2base,scale)
draw_frame(K,T_plat2cam @ T_plat2base @ T_base2hinge,scale)
draw_frame(K,T_plat2cam @ T_plat2base @ T_base2hinge @ T_hinge2arm,scale/2)
draw_frame(K,T_plat2cam @ T_plat2base @ T_base2hinge @ T_hinge2arm @ T_arm2rotors,scale)


u,v = project(K,T_plat2cam @ T_plat2base @ T_base2hinge @ T_hinge2arm @ heli_points[:3,:].T)
plt.scatter(u, v, c='r', marker='.', s=80)

u,v = project(K,T_plat2cam @ T_plat2base @ T_base2hinge @ T_hinge2arm @ T_arm2rotors @ heli_points[3:7,:].T)
plt.scatter(u, v, c='r', marker='.', s=80)

plt.title('Helicopter model')
#plt.xlim([100, 600])
#plt.ylim([600, 300])
plt.show()