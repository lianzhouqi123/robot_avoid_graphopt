from developsuit.assets.bodies.obstacle.obstacle_cylinder_nomove import Obstacle
import numpy as np
import numpy.random as rdm
import math as m


class Fix_Obs:
    def __init__(self, arena):
        self.arena = arena
        self.obs_num = 3

        self.obstacles = []
        self.obs_sizes = []
        self.obs_poses_pass = []
        self.obs_poses_no_pass = []
        self.obs_poses = np.array([[-1, 1],
                                   [0, 1],
                                   [1, 1],
                                   ], dtype=np.double)
        self.obs_pass = np.array([False, True, False], dtype=bool)
        # self.obs_poses_pass = self.

        # rdm_range = [[rmin, rmax], [hmin, hmax]]
        self.rdm_range_no_pass = np.array([[0.1, 0.20],
                                           [0.25, 0.5], ])
        self.rdm_range_pass = np.array([[0.1, 0.20],
                                        [0.01, 0.22], ])

    def reset(self):
        self.obstacle_detach()

        self.obstacles = [Obstacle(self.arena) for _ in range(self.obs_num)]
        self.obs_sizes = []

    def obstacle_create(self):
        for ii in range(self.obs_num):
            if self.obs_pass[ii]:
                rdm_range = self.rdm_range_pass
            else:
                rdm_range = self.rdm_range_no_pass

            r = rdm.uniform(low=rdm_range[0, 0], high=rdm_range[0, 1])
            h = rdm.uniform(low=rdm_range[1, 0], high=rdm_range[1, 1])
            size = np.array([r, h])
            self.obs_sizes.append(size)
            self.obstacles[ii].obstacle_create(size)

        self.obstacle_set_pos()

    def obstacle_detach(self):
        for obs in self.obstacles:
            obs.body.mjcf_model.detach()

    def obstacle_set_pos(self):
        for ii in range(self.obs_num):
            pos = self.obs_poses[ii]
            self.obstacles[ii].set_pos(pos)
