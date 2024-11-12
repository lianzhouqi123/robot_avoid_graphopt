from developsuit.utils.transform_utils import *
from developsuit.utils.dis_polygon import *
from . import MultiRobotsObsAvoidNomujoco


class Env(MultiRobotsObsAvoidNomujoco):
    def __init__(self, n_robots, show_mode="no_show"):
        super().__init__(n_robots=n_robots, show_mode=show_mode, obstacle_have=True, obstacle_size_mode="random",
                         obstacle_pos_mode="fixed", fix_obs_arena_name=None)
        self.target_board = None
        self.n_robots = n_robots
        self.l_form = 0.75
        self.board_radius = 0.3

        self.flex_size = np.array(
            [self.l_form - self.board_radius - 0.3, self.l_form - self.board_radius + 0.3])
        self.flex_theta = np.array([45 / 180 * m.pi])  # 半角

        self.car_vel_max = np.array([0.01, 0.01])
        self.target_range = np.array([[-1, 1], [3.25, 3.75]])

    def reset(self):
        super().init_env()

        x_min = self.target_range[0, 0]
        y_min = self.target_range[1, 0]
        x_range = self.target_range[0, 1] - x_min
        y_range = self.target_range[1, 1] - y_min
        self.target_board = np.array([x_range, y_range]) * np.random.rand(2) + np.array([x_min, y_min])

    def step(self, action):
        action = action.reshape([self.n_robots, -1])
        self.robots_ctrl(joint_steps=action)

        robots_pos_new = self.robots_pos + action
        board_pos_new, board_theta_new = self.get_board_qpos(robots_pos_new)

        self.physics_step(board_pos=board_pos_new, board_thetaz=board_theta_new)

        return 0

    def get_board_qpos(self, robots_pos):
        board_pos = np.mean(robots_pos, axis=0)
        robots_ref = robots_pos - board_pos
        # 让最大值最小
        theta_robots = clip_q(np.atan2(robots_ref[:, 1], robots_ref[:, 0]) - m.pi / 2 - np.arange(
            self.n_robots) * 2 * m.pi / self.n_robots)
        board_theta = (np.max(theta_robots) + np.min(theta_robots)) / 2

        return board_pos, board_theta

    def test_env(self):
        self.reset()

        T = 500

        angle_step = np.linspace(0, 4 * m.pi, T + 1)
        x1 = 0.1 * np.cos(angle_step)
        y1 = 0.1 * np.sin(angle_step)
        x2 = 0.3 * np.cos(angle_step)
        y2 = - 0.3 * np.sin(angle_step)
        x3 = - 0.2 * np.cos(angle_step)
        y3 = - 0.2 * np.sin(angle_step)
        track_design = np.vstack([x1, y1, x2, y2, x3, y3])

        track = []
        track_board = []
        for ii in range(T):
            action = (track_design[:, ii + 1] - track_design[:, ii]).reshape([-1, 2])
            track.append(self.robots_pos)
            track_board.append(self.board_qpos[0:2])
            self.step(action)

        return np.array(track), np.array(track_board)
