import numpy as np
import numpy.random as rdm
import math as m


class Fix_Obs_nomujoco:
    def __init__(self):
        self.obs_num = 6

        self.obs_sizes = []
        self.obs_poses = np.array([[-1.5, 1],
                                   [0, 1.5],
                                   [1, 1],
                                   [-1, 2],
                                   [0.3, 2.3],
                                   [1.5, 2]
                                   ], dtype=np.double)
        self.obs_pass = np.array([False, True, False, True, False, True], dtype=bool)

        # rdm_range = [[rmin, rmax], [hmin, hmax]]
        self.rdm_size_range_no_pass = np.array([[0.1, 0.25],
                                           [0.275, 0.5], ])
        self.rdm_size_range_pass = np.array([[0.1, 0.25],
                                        [0.05, 0.225], ])

    def reset(self):
        self.obs_sizes = []

    def obstacle_create(self):
        for ii in range(self.obs_num):
            if self.obs_pass[ii]:
                rdm_range = self.rdm_size_range_pass
            else:
                rdm_range = self.rdm_size_range_no_pass

            r = rdm.uniform(low=rdm_range[0, 0], high=rdm_range[0, 1])
            h = rdm.uniform(low=rdm_range[1, 0], high=rdm_range[1, 1])
            size = np.array([r, h])
            self.obs_sizes.append(size)
