import time

import numpy as np
from scipy.linalg import block_diag
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
from FootEnd import FootEnd
# 绘图
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

matplotlib.use('TkAgg')


class ISMPCSolver:
    def __init__(self, tp, tc, dt, t_ave, step_ave, ref_len, theta_max, hc, dz_x, dz_y, da_x, da_y, alpha, beta,
                 zmp_vmax, ds_s=0.255, g=9.806,tail_mode="anticipative"):
        self.tp = tp
        self.tc = tc
        self.dt = dt
        self.t_ave = t_ave
        self.step_ave = step_ave
        self.ref_len = ref_len
        self.theta_max = theta_max
        self.hc = hc
        self.dz_x = dz_x
        self.dz_y = dz_y
        self.da_x = da_x
        self.da_y = da_y
        self.alpha = alpha
        self.beta = beta
        self.zmp_vmax = zmp_vmax
        self.ds_s = ds_s
        self.eta = np.sqrt(g / hc)
        self.tail_mode = tail_mode  
        self.num_p = int(self.tp / self.dt)
        self.num_c = int(self.tc / self.dt)

        # CandidateFootstepGen
        self.time_stamp_sequence = np.array([])
        self.step_x_position = None
        self.step_y_position = None
        self.step_theta = None
        self.ref_footstep_num = 0
        self.control_footstep_num = 0

        self.last_footstep_time_stamp = 0
        self.next_footstep_time_stamp = t_ave

        self.footstep_index_sequence = np.zeros(self.num_p, dtype=int)
        self.alpha_sequence = np.zeros(self.num_p)
        self.theta_sequence = np.zeros(self.num_p)

        # 当前信息
        self.current_foot_x = 0
        self.current_foot_y = 0
        self.current_foot_theta = 0
        self.last_foot_x = 0
        self.last_foot_y = 0
        self.last_foot_theta = 0

        # 双足
        self.right_foot = FootEnd("right", 0,0,0,0, self.ds_s)
        self.left_foot = FootEnd("left", 0,0,0,0, self.ds_s)

        self.stand_foot = 0
        self.current_time = 0
        self.xc = 0
        self.yc = 0
        self.dxc = 0
        self.dyc = 0
        self.cop_x = 0
        self.cop_y = 0

        # 控制量
        self.zmp_dx = None
        self.zmp_dy = None
        self.control_foot_step_x = None
        self.control_foot_step_y = None

        # 参考速度轨迹
        self.ref_vx = None
        self.ref_vy = None
        self.ref_w = None

        # MPC轨迹
        self.foot_x = np.array([])
        self.foot_y = np.array([])
        self.foot_theta = np.array([])
        self.xc_traj = None
        self.yc_traj = None
        self.zmp_x_traj = None
        self.zmp_y_traj = None
        self.dxc_traj = None
        self.dyc_traj = None
        self.left_foot_x_traj = None
        self.left_foot_y_traj = None
        self.right_foot_x_traj = None
        self.right_foot_y_traj = None
        self.left_foot_z_traj = None
        self.right_foot_z_traj = None
        self.left_foot_theta_traj = None
        self.right_foot_theta_traj = None

    def ref_velocity(self, vx, vy, w):
        self.ref_vx = vx
        self.ref_vy = vy
        self.ref_w = w

    def init_foot(self,left_foot_x,left_foot_y,left_foot_z,left_foot_theta,
                  right_foot_x,right_foot_y,right_foot_z,right_foot_theta,
                  stand_foot):
        self.stand_foot = stand_foot
        self.right_foot.set_current_foot(right_foot_x, right_foot_y, right_foot_theta,z = right_foot_z)
        self.left_foot.set_current_foot(left_foot_x, left_foot_y, left_foot_theta,z = left_foot_z)
        if stand_foot == "right":  
            self.current_foot_x = right_foot_x
            self.current_foot_y = right_foot_y
            self.current_foot_theta = right_foot_theta
            self.last_foot_x = left_foot_x
            self.last_foot_y = left_foot_y
            self.last_foot_theta = left_foot_theta
        else:
            self.current_foot_x = left_foot_x
            self.current_foot_y = left_foot_y
            self.current_foot_theta = left_foot_theta
            self.last_foot_x = right_foot_x
            self.last_foot_y = right_foot_y
            self.last_foot_theta = right_foot_theta

    def update_current_state(self, xc, yc, dxc, dyc, cop_x, cop_y,
                             left_x,left_y,left_z,left_theta,
                             right_x,right_y,right_z,right_theta,
                             current_time):
        if current_time >= self.next_footstep_time_stamp:
            print("~~~~~~~~~~ change foot ~~~~~~~~~~~~~~~~")
            self.last_footstep_time_stamp = self.next_footstep_time_stamp
            if self.stand_foot == "right":
                self.left_foot.set_current_foot(left_x, left_y, left_theta,z = left_z)
                self.last_foot_x = self.right_foot.current_foot_x
                self.last_foot_y = self.right_foot.current_foot_y
                self.last_foot_theta = self.right_foot.current_foot_theta
                self.current_foot_theta = self.left_foot.current_foot_theta
                self.current_foot_x = self.left_foot.current_foot_x
                self.current_foot_y = self.left_foot.current_foot_y
            else:
                self.right_foot.set_current_foot(right_x, right_y, right_theta,z = right_z)
                self.last_foot_x = self.left_foot.current_foot_x
                self.last_foot_y = self.left_foot.current_foot_y
                self.last_foot_theta = self.left_foot.current_foot_theta
                self.current_foot_theta = self.right_foot.current_foot_theta
                self.current_foot_x = self.right_foot.current_foot_x
                self.current_foot_y = self.right_foot.current_foot_y
            self.stand_foot = "left" if self.stand_foot == "right" else "right"
            # log
            self.foot_x = np.append(self.foot_x, self.current_foot_x)
            self.foot_y = np.append(self.foot_y, self.current_foot_y)
            self.foot_theta = np.append(self.foot_theta, self.current_foot_theta)
        self.current_time = current_time
        self.xc = xc
        self.yc = yc
        self.dxc = dxc
        self.dyc = dyc
        self.cop_x = cop_x
        self.cop_y = cop_y

    def candidate_timings(self):
        v_ave = self.step_ave / self.t_ave
        t = self.last_footstep_time_stamp
        while t < self.tp + self.current_time:
            i = int(t / self.dt)
            v = np.sqrt(self.ref_vx[i] ** 2 + self.ref_vy[i] ** 2)
            ts = self.t_ave * (self.alpha + v_ave) / (self.alpha + v)
            t += ts
            if self.current_time <= t < self.tp + self.current_time:
                self.time_stamp_sequence = np.append(self.time_stamp_sequence, t)
        self.ref_footstep_num = len(self.time_stamp_sequence)
        self.next_footstep_time_stamp = self.time_stamp_sequence[0]
        for i in range(self.ref_footstep_num):
            if self.time_stamp_sequence[i]  <=  self.tc +  self.current_time:
                self.control_footstep_num += 1

    def integral_world_theta(self, t_start, t_end):
        theta = 0
        t = t_start
        i = int(t / self.dt)
        while t < t_end:
            theta += self.ref_w[i] * self.dt
            i += 1
            t += self.dt
        return theta

    def integral_world_velocity(self, t_start, t_end, theta_now):
        x = 0
        y = 0
        theta = theta_now
        t = t_start
        i = int(t / self.dt)
        while t < t_end:
            v_x = np.cos(theta) * self.ref_vx[i] - np.sin(theta) * self.ref_vy[i]
            v_y = np.sin(theta) * self.ref_vx[i] + np.cos(theta) * self.ref_vy[i]
            x += v_x * self.dt
            y += v_y * self.dt
            theta += self.ref_w[i] * self.dt
            i += 1
            t += self.dt
        return [x, y, theta]

    def gen_candidate_footstep_theta(self):
        delta_theta = np.zeros(self.ref_footstep_num)
        for i in range(self.ref_footstep_num):
            if i == 0:
                delta_theta[i] = self.integral_world_theta(self.last_footstep_time_stamp, self.time_stamp_sequence[i])
            else:
                delta_theta[i] = self.integral_world_theta(self.time_stamp_sequence[i - 1], self.time_stamp_sequence[i])
        # P矩阵
        a = np.ones(self.ref_footstep_num) * 4
        a[self.ref_footstep_num - 1] = 2
        P = np.diag(a) + np.diag(-np.ones(self.ref_footstep_num - 1) * 2, 1) + np.diag(
            -np.ones(self.ref_footstep_num - 1) * 2, -1)
        # q,h向量
        q = np.zeros(self.ref_footstep_num)
        h = np.zeros(self.ref_footstep_num * 2)
        for i in range(self.ref_footstep_num):
            if i == 0:
                q[i] = (delta_theta[i + 1] - delta_theta[i] - self.current_foot_theta) * 2
                h[i] = self.current_foot_theta + self.theta_max
                h[i + self.ref_footstep_num] = -self.current_foot_theta + self.theta_max
            elif i == self.ref_footstep_num - 1:
                q[i] = -delta_theta[i] * 2
                h[i] = self.theta_max
                h[i + self.ref_footstep_num] = self.theta_max
            else:
                q[i] = (delta_theta[i + 1] - delta_theta[i]) * 2
                h[i] = self.theta_max
                h[i + self.ref_footstep_num] = self.theta_max
        # G矩阵
        g1 = np.eye(self.ref_footstep_num) + np.eye(self.ref_footstep_num, k=-1) * -1
        g2 = np.eye(self.ref_footstep_num) * -1 + np.eye(self.ref_footstep_num, k=-1)
        G = np.concatenate((g1, g2), axis=0)
        P = csc_matrix(P)
        G = csc_matrix(G)
        self.step_theta = solve_qp(P, q, G=G, h=h, solver='clarabel')
        if self.step_theta is None:
            print("Theta QP solve failed")
            exit(1)

    def gen_signal(self):
        if self.stand_foot == 0 or self.stand_foot == "right":
            return 1
        else:
            return -1

    def gen_candidate_footstep_position(self):
        delta_x = np.zeros(self.ref_footstep_num)
        delta_y = np.zeros(self.ref_footstep_num)
        signal = self.gen_signal()
        theta_now = 0
        for i in range(self.ref_footstep_num):
            if i == 0:
                [deltax, deltay, theta_now] = self.integral_world_velocity(self.last_footstep_time_stamp,
                                                                           self.time_stamp_sequence[i],
                                                                           self.current_foot_theta)
                delta_x[i] = deltax + signal * self.ref_len * (-np.sin(self.step_theta[i]))
                delta_y[i] = deltay + signal * self.ref_len * (np.cos(self.step_theta[i]))
            else:
                [deltax, deltay, theta_now] = self.integral_world_velocity(self.time_stamp_sequence[i - 1],
                                                                           self.time_stamp_sequence[i], theta_now)
                delta_x[i] = deltax + signal * self.ref_len * (-np.sin(self.step_theta[i]))

                delta_y[i] = deltay + signal * self.ref_len * (np.cos(self.step_theta[i]))
            signal *= -1
        # P矩阵
        a = np.ones(self.ref_footstep_num) * 4
        a[self.ref_footstep_num - 1] = 2
        p_temp = np.diag(a) + np.diag(-np.ones(self.ref_footstep_num - 1) * 2, 1) + np.diag(
            -np.ones(self.ref_footstep_num - 1) * 2, -1)
        P = block_diag(p_temp, p_temp)
        P = csc_matrix(P)

        # q,h向量
        q = np.zeros(self.ref_footstep_num * 2)
        h = np.zeros(self.ref_footstep_num * 4)
        for i in range(self.ref_footstep_num):
            if i == 0:
                q[i] = (delta_x[i + 1] - delta_x[i] - self.current_foot_x) * 2
                q[i + self.ref_footstep_num] = (delta_y[i + 1] - delta_y[i] - self.current_foot_y) * 2
            elif i == self.ref_footstep_num - 1:
                q[i] = -delta_x[i] * 2
                q[i + self.ref_footstep_num] = -delta_y[i] * 2
            else:
                q[i] = (delta_x[i + 1] - delta_x[i]) * 2
                q[i + self.ref_footstep_num] = (delta_y[i + 1] - delta_y[i]) * 2
        q = q.reshape((self.ref_footstep_num * 2, 1))
        signal = self.gen_signal()
        for i in range(self.ref_footstep_num):
            if i == 0:
                t1 = np.cos(self.current_foot_theta) * self.current_foot_x + np.sin(
                    self.current_foot_theta) * self.current_foot_y
                t2 = -np.sin(self.current_foot_theta) * self.current_foot_x + np.cos(
                    self.current_foot_theta) * self.current_foot_y
                h[i] = t1 + 0.5 * self.da_x
                h[i + self.ref_footstep_num] = t2 + signal * self.ref_len + 0.5 * self.da_y
                h[i + self.ref_footstep_num * 2] = 0.5 * self.da_x - t1
                h[i + self.ref_footstep_num * 3] = 0.5 * self.da_y - t2 - signal * self.ref_len
            else:
                h[i] = 0.5 * self.da_x
                h[i + self.ref_footstep_num] = signal * self.ref_len + 0.5 * self.da_y
                h[i + self.ref_footstep_num * 2] = 0.5 * self.da_x
                h[i + self.ref_footstep_num * 3] = 0.5 * self.da_y - signal * self.ref_len
            signal *= -1

        # G矩阵
        G = np.zeros((self.ref_footstep_num * 2, self.ref_footstep_num * 2))
        for i in range(self.ref_footstep_num):
            if i == 0:
                G[i, i] = np.cos(self.current_foot_theta)
                G[i, i + self.ref_footstep_num] = np.sin(self.current_foot_theta)
                G[i + self.ref_footstep_num, i] = -np.sin(self.current_foot_theta)
                G[i + self.ref_footstep_num, i + self.ref_footstep_num] = np.cos(self.current_foot_theta)
            else:
                G[i, i - 1] = -np.cos(self.step_theta[i - 1])
                G[i, i] = np.cos(self.step_theta[i - 1])
                G[i, i - 1 + self.ref_footstep_num] = -np.sin(self.step_theta[i - 1])
                G[i, i + self.ref_footstep_num] = np.sin(self.step_theta[i - 1])
                G[i + self.ref_footstep_num, i - 1] = np.sin(self.step_theta[i - 1])
                G[i + self.ref_footstep_num, i] = -np.sin(self.step_theta[i - 1])
                G[i + self.ref_footstep_num, i - 1 + self.ref_footstep_num] = -np.cos(self.step_theta[i - 1])
                G[i + self.ref_footstep_num, i + self.ref_footstep_num] = np.cos(self.step_theta[i - 1])
        G = np.concatenate((G, -G), axis=0)
        G = csc_matrix(G)
        position = solve_qp(P, q, G=G, h=h, solver='clarabel')
        if position is None:
            print("position QP solve failed")
            exit(1)
        self.step_x_position = position[:self.ref_footstep_num]
        self.step_y_position = position[self.ref_footstep_num:]

    def alpha_t(self, t, t_start, t_end):
        return (t - t_start) / ((t_end - t_start) * self.ds_s)
    
    def gen_time_sequence_info(self):
        current_theta = self.current_foot_theta
        last_theta = self.last_foot_theta
        t_start = self.last_footstep_time_stamp
        t_next = self.time_stamp_sequence[0]
        t = self.current_time
        j = 0
        for i in range(self.num_p):
            if j < self.ref_footstep_num:
                if t_start <= t < t_start + (t_next - t_start) * self.ds_s:
                    a = self.alpha_t(t, t_start, t_next)
                    self.alpha_sequence[i] = a
                    self.theta_sequence[i] = (1 - a) * last_theta + a * current_theta
                    self.footstep_index_sequence[i] = j - 1
                elif t_start + (t_next - t_start) * self.ds_s <= t < t_next:
                    self.alpha_sequence[i] = 1
                    self.theta_sequence[i] = current_theta
                    self.footstep_index_sequence[i] = j - 1
                t += self.dt
                if t >= t_next:
                    j += 1
                    if j < self.ref_footstep_num:
                        last_theta = current_theta
                        current_theta = self.step_theta[j - 1]
                        t_start = t_next
                        t_next = self.time_stamp_sequence[j]
                    else:
                        last_theta = current_theta
                        current_theta = self.step_theta[j - 1]
                        t_start = t_next
            else:
                self.alpha_sequence[i] = 0
                self.theta_sequence[i] = current_theta
                self.footstep_index_sequence[i] = j - 1

    def gen_cost_function(self):
        # P矩阵
        P1 = np.eye(self.num_c * 2) * 2
        P2 = np.eye(self.control_footstep_num * 2) * 2 * self.beta
        P = block_diag(P1, P2)

        # q向量
        q = np.zeros(self.num_c * 2 + self.control_footstep_num * 2)
        for i in range(self.control_footstep_num):
            q[self.num_c * 2 + i] = -2 * self.beta * self.step_x_position[i]
            q[self.num_c * 2 + self.control_footstep_num + i] = -2 * self.beta * self.step_y_position[i]
        return csc_matrix(P), q

    def zmp_position_constraint(self):
        h = np.zeros((self.num_c * 4, 1))
        G = np.zeros((self.num_c * 4, self.num_c * 2 + self.control_footstep_num * 2))
        p = np.tril(np.ones((self.num_c, self.num_c)))  # 下三角矩阵
        for i in range(self.num_c):
            rotate_matrix = np.array([[np.cos(self.theta_sequence[i]), -np.sin(self.theta_sequence[i])],
                                      [np.sin(self.theta_sequence[i]), np.cos(self.theta_sequence[i])]])
            if self.footstep_index_sequence[i] == -1:
                temp_h = np.dot(rotate_matrix.T,
                                np.array([(1 - self.alpha_sequence[i]) * self.last_foot_x + self.alpha_sequence[
                                    i] * self.current_foot_x - self.cop_x,
                                          (1 - self.alpha_sequence[i]) * self.last_foot_y + self.alpha_sequence[
                                              i] * self.current_foot_y - self.cop_y]).reshape(2, 1))
                h[i] = 0.5 * self.dz_x + temp_h[0]
                h[i + self.num_c] = 0.5 * self.dz_y + temp_h[1]
                h[i + self.num_c * 2] = 0.5 * self.dz_x - temp_h[0]
                h[i + self.num_c * 3] = 0.5 * self.dz_y - temp_h[1]
                row1 = p[i, :] * self.dt
                row2 = np.zeros((1, self.control_footstep_num))
                temp = np.concatenate((block_diag(row1, row1), block_diag(row2, row2)), axis=1)
                y = np.dot(rotate_matrix.T, temp)
                G[i, :] = y[0, :]
                G[i + self.num_c, :] = y[1, :]
                G[i + self.num_c * 2, :] = -y[0, :]
                G[i + self.num_c * 3, :] = -y[1, :]
            elif self.footstep_index_sequence[i] == 0:
                temp_h = np.dot(rotate_matrix.T,
                                np.array([(1 - self.alpha_sequence[i]) * self.current_foot_x - self.cop_x,
                                          (1 - self.alpha_sequence[i]) * self.current_foot_y - self.cop_y]).reshape(2,
                                                                                                                    1))
                h[i] = 0.5 * self.dz_x + temp_h[0]
                h[i + self.num_c] = 0.5 * self.dz_y + temp_h[1]
                h[i + self.num_c * 2] = 0.5 * self.dz_x - temp_h[0]
                h[i + self.num_c * 3] = 0.5 * self.dz_y - temp_h[1]
                row1 = p[i, :] * self.dt
                row2 = np.zeros((1, self.control_footstep_num))
                row2[0, self.footstep_index_sequence[i]] = -self.alpha_sequence[i]
                temp = np.concatenate((block_diag(row1, row1), block_diag(row2, row2)), axis=1)
                y = np.dot(rotate_matrix.T, temp)
                G[i, :] = y[0, :]
                G[i + self.num_c, :] = y[1, :]
                G[i + self.num_c * 2, :] = -y[0, :]
                G[i + self.num_c * 3, :] = -y[1, :]
            else:
                temp_h = np.dot(rotate_matrix.T,
                                np.array([- self.cop_x, - self.cop_y]).reshape(2, 1))
                h[i] = 0.5 * self.dz_x + temp_h[0]
                h[i + self.num_c] = 0.5 * self.dz_y + temp_h[1]
                h[i + self.num_c * 2] = 0.5 * self.dz_x - temp_h[0]
                h[i + self.num_c * 3] = 0.5 * self.dz_y - temp_h[1]
                row1 = p[i, :] * self.dt
                row2 = np.zeros((1, self.control_footstep_num))
                row2[0, self.footstep_index_sequence[i] - 1] = -1 + self.alpha_sequence[i]
                row2[0, self.footstep_index_sequence[i]] = -self.alpha_sequence[i]
                temp = np.concatenate((block_diag(row1, row1), block_diag(row2, row2)), axis=1)
                y = np.dot(rotate_matrix.T, temp)
                G[i, :] = y[0, :]
                G[i + self.num_c, :] = y[1, :]
                G[i + self.num_c * 2, :] = -y[0, :]
                G[i + self.num_c * 3, :] = -y[1, :]
        return G, h

    def zmp_velocity_constraint(self):
        ub = np.ones((self.num_c * 2 + self.control_footstep_num * 2, 1)) * self.zmp_vmax
        for i in range(self.control_footstep_num):
            ub[self.num_c * 2 + i] = np.inf
            ub[self.num_c * 2 + self.control_footstep_num + i] = np.inf
        lb = -ub
        return lb, ub

    def kinematic_constraint(self):
        # h向量
        h = np.zeros((self.control_footstep_num * 4, 1))
        signal = self.gen_signal()
        for i in range(self.control_footstep_num):
            if i == 0:
                t1 = np.cos(self.current_foot_theta) * self.current_foot_x + np.sin(
                    self.current_foot_theta) * self.current_foot_y
                t2 = -np.sin(self.current_foot_theta) * self.current_foot_x + np.cos(
                    self.current_foot_theta) * self.current_foot_y
                h[i] = t1 + 0.5 * self.da_x
                h[i + self.control_footstep_num] = t2 + signal * self.ref_len + 0.5 * self.da_y
                h[i + self.control_footstep_num * 2] = 0.5 * self.da_x - t1
                h[i + self.control_footstep_num * 3] = 0.5 * self.da_y - t2 - signal * self.ref_len
            else:
                h[i] = 0.5 * self.da_x
                h[i + self.control_footstep_num] = signal * self.ref_len + 0.5 * self.da_y
                h[i + self.control_footstep_num * 2] = 0.5 * self.da_x
                h[i + self.control_footstep_num * 3] = 0.5 * self.da_y - signal * self.ref_len
            signal *= -1
        # G矩阵
        G = np.zeros((self.control_footstep_num * 2, self.control_footstep_num * 2))
        for i in range(self.control_footstep_num):
            if i == 0:
                G[i, i] = np.cos(self.current_foot_theta)
                G[i, i + self.control_footstep_num] = np.sin(self.current_foot_theta)
                G[i + self.control_footstep_num, i] = -np.sin(self.current_foot_theta)
                G[i + self.control_footstep_num, i + self.control_footstep_num] = np.cos(self.current_foot_theta)
            else:
                G[i, i - 1] = -np.cos(self.step_theta[i - 1])
                G[i, i] = np.cos(self.step_theta[i - 1])
                G[i, i - 1 + self.control_footstep_num] = -np.sin(self.step_theta[i - 1])
                G[i, i + self.control_footstep_num] = np.sin(self.step_theta[i - 1])
                G[i + self.control_footstep_num, i - 1] = np.sin(self.step_theta[i - 1])
                G[i + self.control_footstep_num, i] = -np.sin(self.step_theta[i - 1])
                G[i + self.control_footstep_num, i - 1 + self.control_footstep_num] = -np.cos(self.step_theta[i - 1])
                G[i + self.control_footstep_num, i + self.control_footstep_num] = np.cos(self.step_theta[i - 1])
        G = np.concatenate((G, -G), axis=0)
        G = np.concatenate((np.zeros((self.control_footstep_num * 4, self.num_c * 2)), G), axis=1)
        return G, h

    def stability_constraint(self):
        num = self.num_c * 2 + self.control_footstep_num * 2
        xu = self.xc + self.dxc / self.eta
        yu = self.yc + self.dyc / self.eta
        b = np.zeros((2, 1))
        A = np.zeros((2, num))
        if self.tail_mode == "truncated":
            for i in range(self.num_c):
                A[0, i] = np.exp(-i * self.dt * self.eta)
                A[1, i + self.num_c] = np.exp(-i * self.dt * self.eta)
            b[0] = self.eta / (1 - np.exp(-self.dt * self.eta)) * (xu - self.cop_x)
            b[1] = self.eta / (1 - np.exp(-self.dt * self.eta)) * (yu - self.cop_y)
        elif self.tail_mode == "periodic":
            for i in range(self.num_c):
                A[0, i] = np.exp(-i * self.dt * self.eta)
                A[1, i + self.num_c] = np.exp(-i * self.dt * self.eta)
            b[0] = self.eta / (1 - np.exp(-self.dt * self.eta)) * (xu - self.cop_x)*(1-np.exp(-self.dt * self.eta*self.num_c))
            b[1] = self.eta / (1 - np.exp(-self.dt * self.eta)) * (yu - self.cop_y)*(1-np.exp(-self.dt * self.eta*self.num_c))
        elif self.tail_mode == "anticipative":
            for i in range(self.num_c):
                A[0, i] = np.exp(-i * self.dt * self.eta)
                A[1, i + self.num_c] = np.exp(-(i * self.dt * self.eta))
            for i in range(self.num_c, self.num_p):
                if self.footstep_index_sequence[i] == self.control_footstep_num - 1:
                    A[0, num - 2 - self.control_footstep_num] = np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i - 1] - self.alpha_sequence[i]) / self.dt + A[
                                                                    0, num - 2 - self.control_footstep_num]
                    A[0, num - 1 - self.control_footstep_num] = np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i] - self.alpha_sequence[i - 1]) / self.dt + A[
                                                                    0, num - 1 - self.control_footstep_num]
                    A[1, num - 2] = np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i - 1] - self.alpha_sequence[i]) / self.dt + A[1, num - 1]
                    A[1, num - 1] = np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i] - self.alpha_sequence[i - 1]) / self.dt + A[1, num - 1]
                elif self.footstep_index_sequence[i] == self.control_footstep_num:
                    A[0, num - 1 - self.control_footstep_num] = np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i - 1] - self.alpha_sequence[i]) / self.dt + A[
                                                                    0, num - 1 - self.control_footstep_num]
                    A[1, num - 1] = np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i - 1] - self.alpha_sequence[i]) / self.dt + A[1, num - 1]

                    b[0] = b[0] - np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i] - self.alpha_sequence[i - 1]) / self.dt * self.step_x_position[
                            self.footstep_index_sequence[i]]
                    b[1] = b[1] - np.exp(-i * self.dt * self.eta) * (
                            self.alpha_sequence[i] - self.alpha_sequence[i - 1]) / self.dt * self.step_y_position[
                            self.footstep_index_sequence[i]]
                else:
                    b[0] = b[0] - np.exp(-i * self.dt * self.eta) * (
                            (self.alpha_sequence[i - 1] - self.alpha_sequence[i]) * self.step_x_position[
                        self.footstep_index_sequence[i] - 1] + (self.alpha_sequence[i] - self.alpha_sequence[i - 1]) *
                            self.step_x_position[self.footstep_index_sequence[i]]) / self.dt
                    b[1] = b[1] - np.exp(-i * self.dt * self.eta) * (
                            (self.alpha_sequence[i - 1] - self.alpha_sequence[i]) * self.step_y_position[
                        self.footstep_index_sequence[i] - 1] + (self.alpha_sequence[i] - self.alpha_sequence[i - 1]) *
                            self.step_y_position[self.footstep_index_sequence[i]]) / self.dt
            b[0] = b[0] + self.eta / (1 - np.exp(-self.dt * self.eta)) * (xu - self.cop_x)
            b[1] = b[1] + self.eta / (1 - np.exp(-self.dt * self.eta)) * (yu - self.cop_y)
        return csc_matrix(A), b

    def solve(self):
        self.clear()
        self.candidate_timings()
        self.gen_candidate_footstep_theta()
        self.gen_candidate_footstep_position()
        self.gen_time_sequence_info()
        P, q = self.gen_cost_function()
        G1, h1 = self.zmp_position_constraint()
        G2, h2 = self.kinematic_constraint()
        G = csc_matrix(np.concatenate((G1, G2), axis=0))
        h = np.concatenate((h1, h2), axis=0)
        lb, ub = self.zmp_velocity_constraint()
        A, b = self.stability_constraint()
        ans = solve_qp(P, q, G=G, h=h, A=A, b=b, lb=lb, ub=ub, solver='clarabel')
        if ans is None:
            print("MPC QP solve failed")
            self.print_info()
            exit(1)
        else:
            self.zmp_dx = ans[0:self.num_c]
            self.zmp_dy = ans[self.num_c:self.num_c * 2]
            self.control_foot_step_x = ans[self.num_c * 2:self.num_c * 2 + self.control_footstep_num]
            self.control_foot_step_y = ans[self.num_c * 2 + self.control_footstep_num:]
            if self.stand_foot == "right":
                self.left_foot.set_next_foot(self.control_foot_step_x[0], self.control_foot_step_y[0],
                                             self.step_theta[0], self.left_foot.current_z)
                if self.control_footstep_num > 1:
                    self.right_foot.set_next_foot(self.control_foot_step_x[1], self.control_foot_step_y[1],
                                                  self.step_theta[1], self.right_foot.current_z)
            else:
                self.right_foot.set_next_foot(self.control_foot_step_x[0], self.control_foot_step_y[0],
                                              self.step_theta[0], self.right_foot.current_z)
                if self.control_footstep_num > 1:
                    self.left_foot.set_next_foot(self.control_foot_step_x[1], self.control_foot_step_y[1],
                                             self.step_theta[1], self.left_foot.current_z)

    def update_com(self, t):
        if self.zmp_dx is None:
            return np.array([self.xc,self.dxc,self.cop_x]), np.array([self.yc,self.dyc,self.cop_y])
        delta = t -self.current_time
        cosh = np.cosh(self.eta * delta)
        sinh = np.sinh(self.eta * delta)
        A = np.array([[cosh, sinh / self.eta, 1 - cosh],
                    [self.eta * sinh, cosh, -self.eta * sinh],
                    [0, 0, 1]])
        B = np.array([delta - sinh / self.eta, 1 - cosh, delta]).reshape(3, 1)
        predict_x = np.dot(A, np.array([self.xc, self.dxc, self.cop_x]).reshape(3, 1)) + B * self.zmp_dx[0]
        predict_y = np.dot(A, np.array([self.yc, self.dyc, self.cop_y]).reshape(3, 1)) + B * self.zmp_dy[0]
        return predict_x, predict_y

    def clear(self):
        self.time_stamp_sequence = np.array([])
        self.step_x_position = None
        self.step_y_position = None
        self.step_theta = None
        self.ref_footstep_num = 0
        self.control_footstep_num = 0
        self.zmp_dx = None
        self.zmp_dy = None
        self.control_foot_step_x = None
        self.control_foot_step_y = None

    def runMPC(self,t_end,xc,yc,dxc,dyc,zmp_x,zmp_y,
              left_foot_x,left_foot_y,left_foot_z,left_foot_theta,
                right_foot_x,right_foot_y,right_foot_z,right_foot_theta,
               stand_foot):
        num = int(t_end / self.dt)
        self.foot_x = np.array([])
        self.foot_y = np.array([])
        self.foot_theta = np.array([])
        self.xc_traj = np.zeros(num+1)
        self.dxc_traj = np.zeros(num+1)
        self.yc_traj = np.zeros(num+1)
        self.dyc_traj = np.zeros(num+1)
        self.zmp_x_traj = np.zeros(num+1)
        self.zmp_y_traj = np.zeros(num+1)
        self.left_foot_x_traj = np.zeros(num+1)
        self.left_foot_y_traj = np.zeros(num+1)
        self.left_foot_z_traj = np.zeros(num+1)
        self.left_foot_theta_traj = np.zeros(num+1)
        self.right_foot_x_traj = np.zeros(num+1)
        self.right_foot_y_traj = np.zeros(num+1)
        self.right_foot_z_traj = np.zeros(num+1)
        self.right_foot_theta_traj = np.zeros(num+1)

        # 初始
        self.xc_traj[0] = xc
        self.yc_traj[0] = yc
        self.dxc_traj[0] = dxc
        self.dyc_traj[0] = dyc
        self.zmp_x_traj[0] = zmp_x
        self.zmp_y_traj[0] = zmp_y
        self.left_foot_x_traj[0] = left_foot_x
        self.left_foot_y_traj[0] = left_foot_y
        self.left_foot_z_traj[0] = left_foot_z
        self.left_foot_theta_traj[0] = left_foot_theta
        self.right_foot_x_traj[0] = right_foot_x
        self.right_foot_y_traj[0] = right_foot_y
        self.right_foot_z_traj[0] = right_foot_z
        self.right_foot_theta_traj[0] = right_foot_theta
        
        self.init_foot(left_foot_x, left_foot_y,left_foot_z, left_foot_theta, 
                       right_foot_x, right_foot_y, right_foot_z,right_foot_theta,stand_foot)
        for i in range(num):
            print("---------------",i*self.dt,"---------------")
            self.update_current_state(self.xc_traj[i], self.yc_traj[i], self.dxc_traj[i], self.dyc_traj[i], self.zmp_x_traj[i], self.zmp_y_traj[i],
                                    self.left_foot_x_traj[i], self.left_foot_y_traj[i],0, self.left_foot_theta_traj[i],
                                    self.right_foot_x_traj[i], self.right_foot_y_traj[i], 0,self.right_foot_theta_traj[i],
                                    i*self.dt)
            self.solve()
            predict_x, predict_y = self.update_com((i+1)*self.dt)
            self.xc_traj[i+1] = predict_x[0]
            self.yc_traj[i+1] = predict_y[0]
            self.dxc_traj[i+1] = predict_x[1]
            self.dyc_traj[i+1] = predict_y[1]
            self.zmp_x_traj[i+1] = predict_x[2]
            self.zmp_y_traj[i+1] = predict_y[2]
            left_foot = self.left_foot.get_pose((i+1)*self.dt,self.stand_foot,self.last_footstep_time_stamp,self.next_footstep_time_stamp)
            self.left_foot_x_traj[i+1] = left_foot[0]
            self.left_foot_y_traj[i+1] = left_foot[1]
            self.left_foot_z_traj[i+1] = left_foot[2]
            self.left_foot_theta_traj[i+1] = left_foot[3]
            right_foot = self.right_foot.get_pose((i+1)*self.dt,self.stand_foot,self.last_footstep_time_stamp,self.next_footstep_time_stamp)
            self.right_foot_x_traj[i+1] = right_foot[0]
            self.right_foot_y_traj[i+1] = right_foot[1]
            self.right_foot_z_traj[i+1] = right_foot[2]
            self.right_foot_theta_traj[i+1] = right_foot[3]
        self.plot_mpc(num)
    
    def print_info(self):
        print("self.current_time",self.current_time)
        print("self.last_footstep_time_stamp",self.last_footstep_time_stamp)
        print("self.next_footstep_time_stamp",self.next_footstep_time_stamp)
        print("self.time_stamp_sequence",self.time_stamp_sequence)
        print("self.ref_footstep_num",self.ref_footstep_num)
        print("self.control_footstep_num",self.control_footstep_num)
        print("self.step_x_position",self.step_x_position)
        print("self.step_y_position",self.step_y_position)
        print("self.step_theta",self.step_theta)
        print("self.stand_foot",self.stand_foot)
        print("self.current_foot_x",self.current_foot_x)
        print("self.current_foot_y",self.current_foot_y)
        print("self.last_foot_x",self.last_foot_x)
        print("self.last_foot_y",self.last_foot_y)

    def plot_once(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(self.current_foot_x, self.current_foot_y, color='b')
        plt.scatter(self.last_foot_x, self.last_foot_y,color='b')
        plt.scatter(self.step_x_position, self.step_y_position)
        plt.scatter(self.control_foot_step_x, self.control_foot_step_y)
        plt.scatter(self.cop_x, self.cop_y, color='r')
        plt.scatter(self.xc, self.yc, color='g')
        rectangle = patches.Rectangle((self.current_foot_x - self.dz_x / 2,
                                       self.current_foot_y - self.dz_y / 2), self.dz_x, self.dz_y,
                                      angle=self.current_foot_theta * 180 / np.pi,
                                      rotation_point='center', fill=False, color='r')
        ax.add_patch(rectangle)
        rectangle = patches.Rectangle((self.last_foot_x - self.dz_x / 2,
                                       self.last_foot_y - self.dz_y / 2), self.dz_x, self.dz_y,
                                      angle=self.last_foot_theta * 180 / np.pi,
                                      rotation_point='center', fill=False, color='r')
        ax.add_patch(rectangle)
        for i in range(self.ref_footstep_num):
            rectangle = patches.Rectangle((self.step_x_position[i] - self.dz_x / 2,
                                           self.step_y_position[i] - self.dz_y / 2), self.dz_x, self.dz_y,
                                          angle=self.step_theta[i] * 180 / np.pi, rotation_point='center', fill=False,
                                          color='g')
            ax.add_patch(rectangle)
        if self.zmp_dx is not None:
            for i in range(self.control_footstep_num):
                rectangle = patches.Rectangle((self.control_foot_step_x[i] - self.dz_x / 2,
                                            self.control_foot_step_y[i] - self.dz_y / 2), self.dz_x, self.dz_y,
                                            angle=self.step_theta[i] * 180 / np.pi, rotation_point='center',
                                            fill=False, color='k')
                ax.add_patch(rectangle)
            P = np.tril(np.ones(self.num_c)) * self.dt
            Xz = np.ones(self.num_c) * self.cop_x + np.dot(P, self.zmp_dx)
            Yz = np.ones(self.num_c) * self.cop_y + np.dot(P, self.zmp_dy)
            plt.plot(Xz, Yz)
        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax.plot(self.alpha_sequence)
        ax = fig.add_subplot(312)
        ax.plot(self.theta_sequence)
        ax = fig.add_subplot(313)
        ax.plot(self.footstep_index_sequence)
        plt.show()

    def plot_mpc(self,num):
        # --------------- 2D ----------------
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        ax.plot(self.xc_traj, self.yc_traj)
        ax.plot(self.zmp_x_traj, self.zmp_y_traj)
        plt.legend(["com", "zmp"],prop={'family': 'Times New Roman', 'size': 16})
        plt.yticks(fontproperties='Times New Roman', size=14)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.xlabel("x (m)", fontdict={'family': 'Times New Roman', 'size': 16})
        plt.ylabel("y (m)", fontdict={'family': 'Times New Roman', 'size': 16})
        ax.scatter(self.foot_x, self.foot_y)
        ax.scatter(self.left_foot_x_traj[0], self.left_foot_y_traj[0], color='k')
        ax.scatter(self.right_foot_x_traj[0], self.right_foot_y_traj[0], color='k')
        rectangle = patches.Rectangle((self.left_foot_x_traj[0] - self.dz_x / 2,
                                       self.left_foot_y_traj[0] - self.dz_y / 2), self.dz_x, self.dz_y,
                                      angle=self.left_foot_theta_traj[0] * 180 / np.pi,
                                      rotation_point='center', fill=False, color='g')
        ax.add_patch(rectangle)
        rectangle = patches.Rectangle((self.right_foot_x_traj[0] - self.dz_x / 2,
                                        self.right_foot_y_traj[0] - self.dz_y / 2), self.dz_x, self.dz_y,
                                          angle=self.right_foot_theta_traj[0] * 180 / np.pi,
                                          rotation_point='center', fill=False, color='g')
        ax.add_patch(rectangle)
        for i in range(len(self.foot_x)):
            color = ['b','r']
            rectangle = patches.Rectangle((self.foot_x[i] - self.dz_x / 2,
                                        self.foot_y[i] - self.dz_y / 2), self.dz_x, self.dz_y,
                                        angle=self.foot_theta[i] * 180 / np.pi, rotation_point='center',
                                        fill=False, color=color[i%2])
            ax.add_patch(rectangle)
        # -------------------------------
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        t = np.arange(0, num) * self.dt
        ax.plot(t,self.ref_vx[:num])
        ax.plot(t,self.ref_vy[:num])
        ax.plot(t,self.dxc_traj[:num])
        ax.plot(t,self.dyc_traj[:num])
        plt.legend(["vx", "vy", "dxc", "dyc"],prop={'family': 'Times New Roman', 'size': 16})
        plt.xlabel("t (s)", fontdict={'family': 'Times New Roman', 'size': 16})
        plt.ylabel("v (m/s)", fontdict={'family': 'Times New Roman', 'size': 16})
        plt.yticks(fontproperties='Times New Roman', size=14)
        plt.xticks(fontproperties='Times New Roman', size=14)
        # -------------- 3D -----------------
        max_x = max(max(self.left_foot_x_traj),max(self.right_foot_x_traj))
        min_x = min(min(self.left_foot_x_traj),min(self.right_foot_x_traj))
        max_y = max(max(self.left_foot_y_traj),max(self.right_foot_y_traj))
        min_y = min(min(self.left_foot_y_traj),min(self.right_foot_y_traj))
        fig = plt.figure()
        ax2 = plt.axes(projection='3d')
        com = np.array([self.xc_traj,self.yc_traj,np.ones(num+1)*self.hc])
        ax2.set_xlim(min_x-0.1,max_x+0.1)
        ax2.set_ylim(min_y-0.1,max_y+0.1)
        ax2.plot3D(self.left_foot_x_traj,self.left_foot_y_traj,self.left_foot_z_traj,'red')
        ax2.plot3D(self.right_foot_x_traj,self.right_foot_y_traj,self.right_foot_z_traj,'blue')
        ax2.plot3D(com[0,:],com[1,:],com[2,:],'green')
        plt.grid()
        # -----------------------------------
        fig = plt.figure()
        t = 2
        n = int(t/self.dt)
        ax3 = plt.axes(projection='3d')
        ax3.set_xlim(min_x-0.1,max_x+0.1)
        ax3.set_ylim(min_y-0.1,max_y+0.1)
        line1, = ax3.plot3D([], [], [],'red',animated=True)
        line2, = ax3.plot3D([], [], [],'blue',animated=True)
        line3, = ax3.plot3D([], [], [],'green',animated=True)
        def animate(i):
            if i > n:
                line1.set_xdata(self.left_foot_x_traj[i+1-n:i+1])
                line1.set_ydata(self.left_foot_y_traj[i+1-n:i+1])
                line1.set_3d_properties(self.left_foot_z_traj[i+1-n:i+1])
                line2.set_xdata(self.right_foot_x_traj[i+1-n:i+1])
                line2.set_ydata(self.right_foot_y_traj[i+1-n:i+1])
                line2.set_3d_properties(self.right_foot_z_traj[i+1-n:i+1])
                line3.set_xdata(com[0,i+1-n:i+1])
                line3.set_ydata(com[1,i+1-n:i+1])
                line3.set_3d_properties(com[2,i+1-n:i+1])
            else:
                line1.set_xdata(self.left_foot_x_traj[:i+1])
                line1.set_ydata(self.left_foot_y_traj[:i+1])
                line1.set_3d_properties(self.left_foot_z_traj[:i+1])
                line2.set_xdata(self.right_foot_x_traj[:i+1])
                line2.set_ydata(self.right_foot_y_traj[:i+1])
                line2.set_3d_properties(self.right_foot_z_traj[:i+1])
                line3.set_xdata(com[0,:i+1])
                line3.set_ydata(com[1,:i+1])
                line3.set_3d_properties(com[2,:i+1])
            return line1,line2,line3
        frames = len(self.left_foot_x_traj)-1
        ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100,repeat=True,blit = True)
        ani.save('test.gif', writer='pillow', fps=10)
        plt.show()
     
    