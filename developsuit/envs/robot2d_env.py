from developsuit.controllers.smpl_car_controller import Car_Controller
from developsuit.assets.robots import Robot2d
from developsuit.utils.kine import *
from developsuit.utils.mujoco_utils import *


# 二维机器人

class Robot2d_Env:
    def __init__(self):
        # 创建机器人
        self.robot = Robot2d()
        self.mjcf_model = self.robot.mjcf_model

        # 整合关节
        self.joints = self.robot.joints
        self.num_joints = len(self.joints)

        self.physics = None
        self._timestep = None
        self.car_controller = None

        # 控制给入标志
        self.car_control_flag = False

    def create_controller(self, physics):
        self.physics = physics
        # 车控制器
        self.car_controller = Car_Controller(
            physics=self.physics,
            joints=self.joints,
            min_effort=-150.0,
            max_effort=150.0,
            kp=6500,
            damping_ratio=0.32,
            mm=1,
        )
        self._timestep = self.physics.model.opt.timestep

    # 初始化环境
    def init_env(self, car_pos=None):
        # car_pos = [x, y]
        if car_pos is None:
            car_pos = np.array([0., 0.])
        car_pos = car_pos.reshape([-1])
        q_init = car_pos.copy()
        self.init_joint_set(init_joint=q_init, unit="rad")

        # 初始化控制标志
        self.car_control_flag = False

    def ctrl(self, joint_car_step=None, joint_car=None, max_car_sec=None):
        # 关节角控制，step值为单步控制量，无step为绝对量。
        # 同时给一个物体单步和绝对，则以单步为准。
        if max_car_sec is None:
            max_car_sec = np.array([1, 1])
        max_car_step = (max_car_sec * self._timestep).reshape([2])
        if joint_car_step is not None:
            self.car_controller.run(joint_car_step)
            self.car_control_flag = True
        else:
            if joint_car is not None:
                joint_car_step = joint_car - self.pos
                joint_car_step = np.clip(joint_car_step, -max_car_step, max_car_step)
                self.car_controller.run(joint_car_step)
                self.car_control_flag = True

    # 总控，执行物理模型
    def step(self):
        # 如果没被控过，则全给0
        if not self.car_control_flag:
            joint_car_step = np.zeros(self.num_joints)
            self.car_controller.run(joint_car_step)

        # 被控标志清零
        self.car_control_flag = False

    def init_joint_set(self, init_joint, unit: str):
        if unit == "deg":
            for i in range(self.num_joints):
                init_joint[i] = init_joint[i] * m.pi / 180
        if unit == "rad":
            for i in range(self.num_joints):
                init_joint[i] = init_joint[i] * 1

        self.physics.bind(self.joints).qpos[:] = init_joint

    @property
    def pos(self):
        return np.array(self.physics.bind(self.robot.joints).qpos).reshape([-1])

    @property
    def qpos(self):
        return np.array(self.physics.bind(self.robot.joints).qpos).reshape([-1])

    @property
    def qvel(self):
        return np.array(self.physics.bind(self.robot.joints).qvel).reshape([-1])

    # geom编号
    @property
    def geom_id(self):
        return np.array([self.physics.bind(self.robot.geoms).element_id]).reshape([-1])

    @property
    def geom_axis_id(self):
        return np.array([self.physics.bind(self.robot.geoms_axis).element_id]).reshape([-1])
