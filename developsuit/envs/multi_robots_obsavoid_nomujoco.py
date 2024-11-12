import numpy as np
import math as m
from developsuit.arenas import Rdm_Obs_nomujoco
from developsuit.arenas import Fix_Obs_nomujoco


class MultiRobotsObsAvoidNomujoco:
    def __init__(self, n_robots, show_mode="no_show", obstacle_have=False, obstacle_size_mode="fixed",
                 obstacle_pos_mode="fixed", fix_obs_arena_name=None):
        self.board_qpos = None
        self.show_mode = show_mode
        self.obstacle_have = obstacle_have
        self.obstacle_size_mode = obstacle_size_mode
        self.obstacle_pos_mode = obstacle_pos_mode
        self.timestep = 0.01

        self.n_robots = n_robots
        # 创建机械臂
        self.robots = [Robot2d(self.timestep, np.array([0.1])) for _ in range(self.n_robots)]

        self.l_form = 0.75

        # 障碍物
        if self.obstacle_have:
            if self.obstacle_pos_mode == "fixed":
                if fix_obs_arena_name is None:
                    self.obstacle_arena = Fix_Obs_nomujoco()
                else:
                    pass
                    # obs_arena = globals()[fix_obs_arena_name]
                    # self.obstacle_arena = obs_arena(self._arena)
            else:
                self.obstacle_arena = Rdm_Obs_nomujoco(obs_num="Rdm")
            self.obstacle_arena.reset()
            self.obstacle_arena.obstacle_create()

        self.car_vel_max = np.array([0.02, 0.02])
        self.action_max = self.car_vel_max

    # 初始化环境
    def init_env(self):
        # 初始末端位置，距中心0.7的正三角形
        car_init = np.array([
            [self.l_form * m.cos(2 * m.pi / self.n_robots * ii + m.pi / 2),
             self.l_form * m.sin(2 * m.pi / self.n_robots * ii + m.pi / 2)]
            for ii in range(self.n_robots)
        ])

        # 初始化板
        self.board_qpos = np.array([0., 0., 0.], dtype=np.float32)

        # 初始化机器人
        for ii in range(self.n_robots):
            self.robots[ii].reset(pos=car_init[ii, :])

        # 更改障碍物尺寸
        if self.obstacle_have:
            if self.obstacle_size_mode != "fixed":
                self.obstacle_arena.reset()
                self.obstacle_arena.obstacle_create()

                # 放置障碍物
                if self.obstacle_pos_mode != "fixed":
                    self.obstacle_arena.obstacle_set_pos()

    def robots_ctrl(self, joint_steps=None):
        if joint_steps is not None:
            for ii in range(self.n_robots):
                joint_step = joint_steps[ii, :]
                self.robots[ii].ctrl(joint_car_step=joint_step)

    # 动板子，让板子位于多机器人的中心，指向第一个机器人
    def board_move(self, pos2d=None, thetaz=None):
        if pos2d is None:
            pos2d = np.mean(self.robots_pos, axis=0)
        if thetaz is None:
            yline = self.robots[0].pos.reshape([-1]) - pos2d
            thetaz = m.atan2(yline[1], yline[0])
            thetaz -= m.pi / 2
        else:
            thetaz = thetaz.reshape([])
        self.board_qpos[0:2] = pos2d.reshape([-1])
        self.board_qpos[2] = thetaz

    # 总控，执行物理模型
    def physics_step(self, board_pos=None, board_thetaz=None):
        for ii in range(self.n_robots):
            self.robots[ii].step()

        self.board_move(pos2d=board_pos, thetaz=board_thetaz)

    @property
    def robots_pos(self):
        robots_pos = []
        for ii in range(self.n_robots):
            robots_pos.append(self.robots[ii].pos)
        robots_pos = np.vstack(robots_pos)

        return robots_pos

    @property
    def robots_vel(self):
        robots_vel = []
        for ii in range(self.n_robots):
            robots_vel.append(self.robots[ii].vel)
        robots_vel = np.vstack(robots_vel)

        return robots_vel

    @property
    def robots_size(self):
        robots_size = []
        for ii in range(self.n_robots):
            robots_size.append(self.robots[ii].size)
        robots_size = np.vstack(robots_size)

        return robots_size

    # 障碍物位置
    @property
    def obstacle_pos(self):
        return np.array(self.obstacle_arena.obs_poses)

    # 障碍物尺寸
    @property
    def obstacle_size(self):
        return np.array(self.obstacle_arena.obs_sizes)


class Robot2d:
    def __init__(self, timestep, radius):
        self.vel_temp = None
        self.pos_temp = None
        self.pos = None
        self.vel = np.array([0., 0.])
        self.acc = np.array([0., 0.])
        self.kp = 6500
        self.kd = 2 * np.sqrt(self.kp) * 0.32
        self.timestep = timestep
        self.size = radius.reshape([-1])

    def ctrl(self, joint_car_step):
        joint_car_step = joint_car_step.reshape([-1])

        self.acc = self.kp * joint_car_step - self.kd * self.vel

        self.vel_temp = self.vel + self.acc * self.timestep
        self.pos_temp = self.pos + self.vel * self.timestep + 0.5 * self.acc * self.timestep ** 2

    def step(self):
        if self.pos_temp is not None:
            self.pos = self.pos_temp.copy()
        if self.vel_temp is not None:
            self.vel = self.vel_temp.copy()

        self.pos_temp, self.vel_temp = None, None

    def reset(self, pos=None, vel=None, acc=None):
        if pos is None:
            self.pos = np.array([0., 0.])
        else:
            self.pos = pos
        if vel is None:
            self.vel = np.array([0., 0.])
        else:
            self.vel = vel
        if acc is None:
            self.acc = np.array([0., 0.])
        else:
            self.acc = acc

        self.pos_temp, self.vel_temp = None, None
