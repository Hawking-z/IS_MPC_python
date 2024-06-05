import numpy as np
from splines import QuinticHermiteSplines


class FootEnd:
    def __init__(self, name, init_x, init_y,init_z, init_theta, ds_s, step_height=0.1):
        self.current_foot_x = init_x
        self.current_foot_y = init_y
        self.current_foot_theta = init_theta
        self.name = name
        self.ds_s = ds_s
        self.next_foot_x = 0
        self.next_foot_y = 0
        self.next_foot_theta = 0
        self.current_z = init_z
        self.next_z = 0
        self.step_height = step_height
        self.traj_x = QuinticHermiteSplines()
        self.traj_y = QuinticHermiteSplines()
        self.traj_theta = QuinticHermiteSplines()
        self.traj_z_first_chunk = QuinticHermiteSplines()
        self.traj_z_second_chunk = QuinticHermiteSplines()

    def set_current_foot(self, x, y, theta, z=0.0):
        self.current_foot_x = x
        self.current_foot_y = y
        self.current_foot_theta = theta
        self.current_z = z
        self.traj_x.set_start_point(x)
        self.traj_y.set_start_point(y)
        self.traj_theta.set_start_point(theta)
        self.traj_z_first_chunk.set_start_point(z)

    def set_next_foot(self, x, y, theta, z=0.0):
        self.next_foot_x = x
        self.next_foot_y = y
        self.next_foot_theta = theta
        self.next_z = z
        self.traj_x.set_end_point(x)
        self.traj_y.set_end_point(y)
        self.traj_theta.set_end_point(theta)
        max_z = max(self.current_z, self.next_z)
        self.traj_z_first_chunk.set_end_point(max_z + self.step_height)
        self.traj_z_second_chunk.set_start_point(max_z + self.step_height)
        self.traj_z_second_chunk.set_end_point(self.next_z)
        self.traj_x.solve()
        self.traj_y.solve()
        self.traj_theta.solve()
        self.traj_z_first_chunk.solve()
        self.traj_z_second_chunk.solve()

    def get_pose(self, t,stand_foot,last_stamp,next_stamp):
        take_off_start = (next_stamp-last_stamp)*self.ds_s+last_stamp
        if self.name == stand_foot:
            return np.array([self.current_foot_x, self.current_foot_y, self.current_z,self.current_foot_theta],dtype=np.float32)
        else:
            if t < take_off_start:
                return np.array([self.current_foot_x, self.current_foot_y, self.current_z,self.current_foot_theta],dtype=np.float32)
            elif take_off_start <= t < next_stamp: 
                duration = (next_stamp-last_stamp)*(1-self.ds_s)
                foot_x = self.traj_x.get_point((t - take_off_start) / duration)
                foot_y = self.traj_y.get_point((t - take_off_start) / duration)
                foot_theta = self.traj_theta.get_point((t - take_off_start) / duration)
                if  t <  duration / 2 + take_off_start:
                    foot_z = self.traj_z_first_chunk.get_point(
                        ((t - take_off_start) / duration * 2))
                else:
                    foot_z = self.traj_z_second_chunk.get_point(
                        ((t - take_off_start) /duration * 2) - 1)
                return np.array([foot_x, foot_y, foot_z,foot_theta],dtype=np.float32)
            else:
                return np.array([self.next_foot_x, self.next_foot_y, self.next_z,self.next_foot_theta],dtype=np.float32)

    def get_footholds(self,t,stand_foot,last_stamp,next_stamp):
        ts = (next_stamp-last_stamp)
        if self.name == stand_foot:
            return self.current_foot_x,self.current_foot_y,self.current_foot_theta,1,ts*self.ds_s+ts-t+last_stamp
        else:
            if t < (next_stamp-last_stamp)*self.ds_s+last_stamp:
                return self.current_foot_x,self.current_foot_y,self.current_foot_theta,1,ts*self.ds_s-t+last_stamp
            else:
                return self.next_foot_x,self.next_foot_y,self.next_foot_theta,0,ts-(t-last_stamp-ts*self.ds_s)