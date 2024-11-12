import numpy as np
import math as m
from developsuit.utils.mujoco_utils import get_fullM
from developsuit.utils.transform_utils import *


class Car_Controller:
    def __init__(self, physics, joints, min_effort, max_effort, kp, damping_ratio, mm=None):
        self._physics = physics
        self._joints = joints
        self._min_effort = min_effort
        self._max_effort = max_effort
        self._kp = kp
        self._damping_ratio = damping_ratio
        self._kd = 2 * np.sqrt(self._kp) * self._damping_ratio
        self._jnt_dof_ids = self._physics.bind(self._joints).dofadr
        if mm is None:
            self._mm = 1
        else:
            self._mm = mm
        self._timestep = self._physics.model.opt.timestep

    def run(self, target):
        M_full = get_fullM(self._physics.model.ptr, self._physics.data.ptr)
        # 控制转轴
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        dq = self._physics.bind(self._joints).qvel
        target = np.array(target).reshape([-1])
        torque = (self._kp * np.dot(M, target) - self._kd * np.dot(M, dq)) / self._mm
        torque += self._physics.bind(self._joints).qfrc_bias
        # torque = np.clip(torque, self._min_effort, self._max_effort)
        self._physics.bind(self._joints).qfrc_applied = torque
