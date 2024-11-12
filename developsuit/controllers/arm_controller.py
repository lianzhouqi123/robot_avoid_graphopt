import numpy as np
import math
from developsuit.utils.mujoco_utils import (
    get_fullM,
)


class Arm_Controller:
    def __init__(
            self,
            physics: object,
            joints: object,
            min_effort,
            max_effort,
            kp: float,
            damping_ratio,
            mm=None
    ) -> None:
        self._physics = physics
        self._joints = joints
        self._min_effort = min_effort
        self._max_effort = max_effort
        self._kp = kp
        self._damping_ratio = damping_ratio
        self._kd = 2 * np.sqrt(self._kp) * self._damping_ratio
        self._jnt_dof_ids = self._physics.bind(self._joints).dofadr
        if mm is None:
            self.mm = 1
        else:
            self.mm = mm

    def run(self, target) -> None:
        M_full = get_fullM(
            self._physics.model.ptr,
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids[:], :][:, self._jnt_dof_ids[:]]
        dq = self._physics.bind(self._joints).qvel[:]
        ddq = self._physics.bind(self._joints).qacc[:]
        target = np.array(target).reshape([-1])
        torque = (self._kp * np.dot(M, target) - self._kd * np.dot(M, dq)) / self.mm
        torque += self._physics.bind(self._joints).qfrc_bias[:]
        # torque = np.clip(torque, self._min_effort, self._max_effort)
        self._physics.bind(self._joints).qfrc_applied[:] = torque[:]
        pass
