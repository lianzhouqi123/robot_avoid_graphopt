import time
import mujoco.viewer
from dm_control import mjcf
from .robot2d_env import Robot2d_Env
from developsuit.arenas import *
from developsuit.assets.bodies.board_circle.board_circle import Board
from developsuit.controllers.board_controller import Board_Controller
from developsuit.utils.kine import *
from developsuit.utils.mujoco_utils import *


# 机械臂末端安装机械抓

class MultiRobotsObsAvoid:
    def __init__(self, n_robots, show_mode="no_show", obstacle_have=False, obstacle_size_mode="fixed",
                 obstacle_pos_mode="fixed", fix_obs_arena_name=None):
        self.show_mode = show_mode
        self.obstacle_have = obstacle_have
        self.obstacle_size_mode = obstacle_size_mode
        self.obstacle_pos_mode = obstacle_pos_mode
        self._mjcf_model = mjcf.RootElement()
        # 创建地板
        self._arena = Arena()

        self.n_robots = n_robots
        # 创建机械臂
        self.robots = [Robot2d_Env() for _ in range(self.n_robots)]

        # 放机械臂
        for ii in range(self.n_robots):
            self._arena.attach(self.robots[ii].mjcf_model)

        # 创建板子
        self.board_radius = 0.3
        self.board_size = np.array([self.board_radius, 0.02])
        self.board = Board(self._arena, board_size=self.board_size)
        self.board_joint = self.board.joints
        self.board_height = 0.5

        self.l_form = 0.75

        # 障碍物
        if self.obstacle_have:
            if self.obstacle_pos_mode == "fixed":
                if fix_obs_arena_name is None:
                    self.obstacle_arena = Fix_Obs(self._arena)
                else:
                    obs_arena = globals()[fix_obs_arena_name]
                    self.obstacle_arena = obs_arena(self._arena)
            else:
                self.obstacle_arena = Rdm_Obs(self._arena, obs_num="Rdm")
            self.obstacle_arena.reset()
            self.obstacle_arena.obstacle_create()

        # mocap
        self.mocap_t = self._arena.mjcf_model.worldbody.add("body", name="mocap_t", mocap=True)
        self.mocap_t.add(
            "geom",
            type="cylinder",
            size=[0.05, 0.02],
            rgba=[1, 0, 0, 0.2],
            conaffinity=0,
            contype=0,
            pos=[0.0, 0.0, 0.0],  # -0.5, -0.2 + 0.4 * np.random.random(), 0.1
            quat=[1, 0, 0, 0],
        )

        (self.board_controller, self.contact, self._viewer, self._timestep, self.physics, self._step_start,
         self.cnt_permit_id) \
            = None, None, None, None, None, None, None
        self.physics_create()

        self.car_vel_max = np.array([0.02, 0.02])
        self.action_max = self.car_vel_max

    def physics_create(self):
        # 关闭老的物理模型
        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.close()
        if self.physics is not None:
            del self.physics
        # 生成物理模型
        self.physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        self.physics.data.time = 0.01
        self._timestep = self.physics.model.opt.timestep
        self.contact = self.physics.data.contact
        if self.show_mode == "show":
            self._viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
            self._viewer.cam.distance = 8
            # self._viewer.cam.elevation = -45
            self._viewer.cam.elevation = -90
            self._viewer.cam.lookat = [0, 2.25, 0]

        # 控制器
        for ii in range(self.n_robots):
            self.robots[ii].create_controller(self.physics)
        self.board_controller = Board_Controller(
            physics=self.physics,
            joints=self.board_joint,
            min_effort=-150.0,
            max_effort=150.0,
            kp=6500,
            damping_ratio=0.32,
            mm=1,
        )

        # 允许的接触geom的编号对
        self.cnt_permit_id = np.vstack((
            np.vstack([np.hstack([self.ground_geom_id, self.robots[ii].geom_id]) for ii in range(self.n_robots)]),
            np.array([[self.ground_geom_id[0], obstacle_id] for obstacle_id in self.obstacles_id]),
            )).reshape([-1, 2])

    # 初始化环境
    def init_env(self):
        # 初始末端位置，距中心0.75的正三角形
        car_init = np.array([
            [self.l_form * m.cos(2 * m.pi / self.n_robots * ii + m.pi / 2),
             self.l_form * m.sin(2 * m.pi / self.n_robots * ii + m.pi / 2)]
            for ii in range(self.n_robots)
        ])

        # 初始化板
        board_pos_init = np.concatenate((np.array([0., 0., self.board_height]), np.array([1., 0., 0., 0.])))

        # 更改障碍物尺寸
        if self.obstacle_have:
            if self.obstacle_size_mode != "fixed":
                self.obstacle_arena.reset()
                self.obstacle_arena.obstacle_create()

                # 放置障碍物
                if self.obstacle_pos_mode != "fixed":
                    self.obstacle_arena.obstacle_set_pos()

                self.physics_create()

        # 重置物理系统，带入值
        with self.physics.reset_context():
            for ii in range(self.n_robots):
                self.robots[ii].init_env(car_init[ii, :])

            self.physics.bind(self.board_joint).qpos[:] = board_pos_init

        # 显示
        self.render()

    def robots_ctrl(self, joints=None, joint_steps=None):
        joint, joint_step = None, None
        for ii in range(self.n_robots):
            if joints is not None:
                joint = joints[ii, :]
            if joint_steps is not None:
                joint_step = joint_steps[ii, :]
            self.robots[ii].ctrl(joint_car=joint, joint_car_step=joint_step, max_car_sec=self.car_vel_max)

    # 动板子，让板子位于多机器人的中心，指向第一个机器人
    def board_move(self, pos2d=None, thetaz=None):
        if pos2d is None:
            pos2d = np.mean(self.robots_pos, axis=0)
        if thetaz is None:
            yline = self.robots[0].pos.reshape([-1]) - pos2d
            thetaz = m.atan2(yline[1], yline[0])
            thetaz -= m.pi/2
        else:
            thetaz = thetaz.reshape([])
        eul = np.array([0., 0., thetaz])

        eul_curr = quat2eul(self.board_qpos[3:7])
        epsino = m.pi / 2 - 5e-1
        # 判断欧拉角模式，避开奇异点
        if abs(eul_curr[1, 0]) > epsino:
            eul_curr = quat2eul(self.board_qpos[3:7], mode="XZY")

        pos3d = np.hstack((pos2d, np.array([self.board_height])))
        pos_step = pos3d - self.board_qpos[0:3]
        eul_step = eul - eul_curr.reshape([-1])
        joint_step = np.hstack((pos_step, eul_step))
        self.board_controller.run(joint_step)

    # 总控，执行物理模型
    def physics_step(self, board_pos=None, board_thetaz=None):
        for ii in range(self.n_robots):
            self.robots[ii].step()

        self.board_move(pos2d=board_pos, thetaz=board_thetaz)

        # 物理系统执行
        self.physics.step()
        # 显示
        self.render()

    def render(self):
        if self.show_mode == "show":
            # 可视化
            self._step_start = time.time()
            self._viewer.sync()
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    @property
    def robots_pos(self):
        return np.array([self.robots[ii].pos for ii in range(self.n_robots)]).reshape([-1, 2])

    @property
    def robots_vel(self):
        return np.array([self.robots[ii].qvel for ii in range(self.n_robots)]).reshape([-1, 2])

    # 板子的位姿
    @property
    def board_qpos(self):
        return np.array(self.physics.bind(self.board_joint).qpos).reshape([-1])

    # 障碍物位置
    @property
    def obstacle_pos(self):
        return np.array(self.obstacle_arena.obs_poses)

    # 障碍物尺寸
    @property
    def obstacle_size(self):
        return np.array(self.obstacle_arena.obs_sizes)

    # 大地geom编号
    @property
    def ground_geom_id(self):
        return np.array([self.physics.bind(self._arena.ground_geom).element_id]).reshape([-1])

    # 板子geom编号
    @property
    def board_id(self):
        return np.array([self.physics.bind(self.board.geom).element_id]).reshape([-1])

    # 障碍物编号
    @property
    def obstacles_id(self):
        if self.obstacle_have:
            obstacle_id = np.hstack([self.physics.bind(self.obstacle_arena.obstacles[ii].geom).element_id
                                    for ii in range(self.obstacle_arena.obs_num)]).reshape([-1])
        else:
            obstacle_id = np.array([-1])
        return obstacle_id

    # 所有不允许接触的接触序号对
    @property
    def cnt_id(self):
        cnt_id = []
        cnt_id_all = np.array(self.contact.geom).reshape([-1, 2])
        for row in cnt_id_all:
            # 判断每一行是否出现在允许的接触中
            if not np.any(np.all(self.cnt_permit_id == row, axis=1)):
                cnt_id.append(row)

        cnt_id = np.array(cnt_id).reshape([-1, 2])

        return cnt_id

    @property
    def arena(self):
        return self._arena

    @property
    def board_thetaz(self):
        eul_curr = quat2eul(self.board_qpos[3:7])
        epsino = m.pi / 2 - 5e-1
        # 判断欧拉角模式，避开奇异点
        if abs(eul_curr[1, 0]) > epsino:
            eul_curr = quat2eul(self.board_qpos[3:7], mode="XZY")

        thetaz = eul_curr[2].reshape([-1])

        return thetaz