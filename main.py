import numpy as np
from helper import class_to_dict
from ISMPCSolver import ISMPCSolver

class mpc_params:
    dt = 0.02
    alpha = 0.2
    tp = 3.0
    tc = 1.8
    ref_len = 0.17
    t_ave = 0.6
    step_ave = 0.3
    theta_max = np.pi / 8
    hc = 0.65
    dz_x = 0.10
    dz_y = 0.06
    da_x = 0.8
    da_y = 0.16
    beta = 20000
    zmp_vmax = 5
    tail_mode = "anticipative" # "anticipative" , "truncated" , "periodic"
    ds_s = 0.2

class init_state:
    left_foot_x = 0
    left_foot_y = 0.16
    right_foot_x = 0
    right_foot_y = 0
    xc = (left_foot_x + right_foot_x) / 2
    yc = (left_foot_y + right_foot_y) / 2
    dxc = 0
    dyc = 0
    zmp_x = xc
    zmp_y = yc
    left_foot_theta = 0
    right_foot_theta = 0
    left_foot_z = 0
    right_foot_z = 0
    stand_foot = "right"

param = class_to_dict(mpc_params)
init_state = class_to_dict(init_state)
solver = ISMPCSolver(**param)
dt = mpc_params.dt
first = int(4 / dt)
second = int(10 / dt)
# vx = np.concatenate((np.ones(first) * 0.3, np.ones(second) * -0.3))
# vy = np.concatenate((np.ones(first) * 0.2, np.ones(second) * 0.2))
# w = np.ones(first + second) * 0.0
# vx = np.tile(vx, 3)
# vy = np.tile(vy, 3)
# w = np.tile(w, 3)
vx = np.concatenate((np.ones(first) * 0.4, np.ones(second) * 0.4))
vy = np.concatenate((np.ones(first) * 0, np.ones(second) * 0))
w = np.concatenate((np.ones(first) * 0.3, np.ones(second) * -0.3))
solver.ref_velocity(vx, vy, w)
solver.runMPC(10, **init_state,index = 3)



