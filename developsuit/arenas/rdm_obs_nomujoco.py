import numpy as np
import numpy.random as rdm
import math as m


class Rdm_Obs_nomujoco:
    def __init__(self, h_pass=0.25, obs_num=None, rdm_size_range=None, rdm_pos_range=None):
        if obs_num is None or obs_num == "Rdm":
            self.obs_num_flag = "Rdm"
            self.obs_num = None
        else:
            self.obs_num_flag = "Fix"
            self.obs_num = obs_num

        # rdm_range = [[rmin, rmax], [hmin, hmax]]
        if rdm_size_range is None:
            rdm_size_range = np.array([[0.1, 0.25],
                                       [0.05, 0.5], ])
        self.rdm_size_range = rdm_size_range
        self.h_pass = h_pass
        if rdm_pos_range is None:
            rdm_pos_range = np.array([[-2.5, 2.5],
                                      [1, 4], ])
        self.rdm_pos_range = rdm_pos_range
        self.low_num, self.high_num = 3, 6
        self.obs_gap = 0.4  # 初始化障碍物时，障碍物间的最小距离
        self.obs_num_nopass = 4
        self.obs_corner = True
        self.corner_size = 0.75

        self.robots_init_pos = np.array([[0., 0.6],
                                         [-0.6 / 2 * m.sqrt(3), -0.6 / 2],
                                         [0.6 / 2 * m.sqrt(3), -0.6 / 2],
                                         ])

        self.obstacles = []
        self.obs_sizes = []
        self.obs_pass = []
        self.obs_poses = []

    def set_property(self, obs_num=None, obs_num_flag=None, rdm_size_range=None, rdm_pos_range=None,
                     low_num=None, high_num=None, obs_gap=None, obs_num_nopass=None, obs_corner=None):
        if obs_num is not None:
            self.obs_num_flag = "fix"
            self.obs_num = obs_num
        else:
            if obs_num_flag is not None:
                self.obs_num_flag = obs_num_flag

        if rdm_size_range is not None:
            self.rdm_size_range = rdm_size_range

        if rdm_pos_range is not None:
            self.rdm_pos_range = rdm_pos_range

        if low_num is not None:
            self.low_num = low_num

        if high_num is not None:
            self.high_num = high_num

        if obs_gap is not None:
            self.obs_gap = obs_gap

        if obs_num_nopass is not None:
            self.obs_num_nopass = obs_num_nopass

        if obs_corner is not None:
            self.obs_corner = obs_corner

    def reset(self):
        if self.obs_num_flag == "Rdm":
            self.obs_num = rdm.randint(self.low_num, self.high_num)
        if self.obs_corner:
            self.obs_num += 4

        # self.obstacles = [Obstacle() for _ in range(self.obs_num)]
        self.obs_sizes = []
        self.obs_pass = []
        self.obs_poses = []

    def obstacle_create(self, restrict_nopass_num=True):
        rdm_size_range_pass = self.rdm_size_range.copy()
        rdm_size_range_pass[1, 1] = 0.225
        rdm_size_range_nopass = self.rdm_size_range.copy()
        rdm_size_range_nopass[1, 0] = 0.275

        num_nopass = 0
        for ii in range(self.obs_num):
            if restrict_nopass_num:
                if num_nopass < self.obs_num_nopass:
                    rdm_size_range = rdm_size_range_nopass
                    num_nopass += 1
                else:
                    rdm_size_range = rdm_size_range_pass
            else:
                rdm_size_range = self.rdm_size_range

            if self.obs_corner and ii >= self.obs_num - 4:
                rdm_size_range = rdm_size_range_nopass

            r = rdm.uniform(low=rdm_size_range[0, 0], high=rdm_size_range[0, 1])
            h = rdm.uniform(low=rdm_size_range[1, 0], high=rdm_size_range[1, 1])
            size = np.array([r, h])
            self.obs_sizes.append(size)
            if size[1] >= self.h_pass:
                # obs_pass = False
                self.obs_pass.append(False)
            else:
                # obs_pass = True
                self.obs_pass.append(True)
            # self.obstacles[ii].obstacle_create(size, obs_pass)
        self.obs_pass = np.array(self.obs_pass, dtype=bool)

    def obstacle_set_pos(self):
        # rdm_range = [[xmin, xmax], [ymin, ymax]]
        self.obs_poses = []
        obs_sizes = np.vstack(self.obs_sizes)
        for ii in range(self.obs_num):
            if self.obs_corner:
                if ii == self.obs_num - 4:
                    pos = np.array(
                        [self.rdm_pos_range[0, 0] - self.corner_size, self.rdm_pos_range[1, 0] - self.corner_size])
                elif ii == self.obs_num - 3:
                    pos = np.array(
                        [self.rdm_pos_range[0, 0] - self.corner_size, self.rdm_pos_range[1, 1] + self.corner_size])
                elif ii == self.obs_num - 2:
                    pos = np.array(
                        [self.rdm_pos_range[0, 1] + self.corner_size, self.rdm_pos_range[1, 0] - self.corner_size])
                else:
                    pos = np.array(
                        [self.rdm_pos_range[0, 1] + self.corner_size, self.rdm_pos_range[1, 1] + self.corner_size])

            if not self.obs_corner or ii < self.obs_num - 4:
                flag_good_obs = False
                for jj in range(50):
                    pos_x = rdm.uniform(low=self.rdm_pos_range[0, 0], high=self.rdm_pos_range[0, 1])
                    pos_y = rdm.uniform(low=self.rdm_pos_range[1, 0], high=self.rdm_pos_range[1, 1])
                    pos = np.array([pos_x, pos_y])
                    if ii == 0:
                        break
                    else:
                        for _ in range(15):
                            dis_obs = (np.linalg.norm(pos - np.vstack(self.obs_poses), axis=1)
                                       - (obs_sizes[0:ii, 0] + obs_sizes[ii, 0] + self.obs_gap))
                            dis_robots = (np.linalg.norm(pos)
                                          - (0.6 + obs_sizes[ii, 0] + self.obs_gap))

                            if np.min(dis_obs) < 0:
                                index = np.argmin(dis_obs)
                                pos = ((pos - self.obs_poses[index])
                                       / np.linalg.norm(pos - self.obs_poses[index])
                                       * (self.obs_gap + obs_sizes[index, 0] + obs_sizes[ii][0])
                                       + self.obs_poses[index]
                                       + 0.05 * (np.random.rand(2) - np.array([0.5, 0.5])))
                            elif np.min(dis_robots) < 0:
                                flag_good_obs = False
                                break
                                # index = np.argmin(dis_robots)
                                # pos = ((pos - self.robots_init_pos[index, :])
                                #              / np.linalg.norm(pos - self.robots_init_pos[index, :])
                                #              * (self.obs_gap + 0.6 + obs_sizes[ii][0])
                                #              + 0.05 * (np.random.rand(2) - np.array([0.5, 0.5])))
                            elif not ((self.rdm_pos_range[:, 0] < pos).all()
                                      and (pos < self.rdm_pos_range[:, 1]).all()):
                                flag_good_obs = False
                                break
                            else:
                                flag_good_obs = True
                                break

                    if flag_good_obs:
                        break

            self.obs_poses.append(pos)
            # self.obstacles[ii].set_pos(pos)


# class Obstacle:
#     def __init__(self):
#         self.pos = None
#         self.size = None
#         self.obs_pass = None
#
#     def obstacle_create(self, size, obs_pass):
#         self.size = size.copy()
#         self.obs_pass = obs_pass
#
#     def set_pos(self, pos):
#         self.pos = pos.copy()
