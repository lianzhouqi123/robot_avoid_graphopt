import numpy as np
import torch
import time
import math as m
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon


def create_star_points(radius, center, num_points=5, inner_ratio=None):
    if inner_ratio is None:
        inner_ratio = 2 * m.sin(4 * m.pi / num_points) * m.sin(m.pi / num_points) / (1 - m.cos(4 * m.pi / num_points))
        # inner_ratio2 = m.sqrt((m.sin(4 * m.pi / num_points) * (m.cos(2 * m.pi / num_points) - 1) / (
        #             m.cos(4 * m.pi / num_points) - 1)) ** 2 + (m.cos(2 * m.pi / num_points)) ** 2)
    angles_in = np.linspace(0, 2 * m.pi, num_points, endpoint=False) - m.pi / 2
    angles_out = angles_in + m.pi / num_points

    outer_points = np.array([(np.cos(angle), np.sin(angle)) for angle in angles_out]) * radius
    inner_points = np.array([(np.cos(angle), np.sin(angle)) for angle in angles_in]) * radius * inner_ratio

    points = []
    for ii in range(num_points):
        points.append(inner_points[ii])
        points.append(outer_points[ii])

    return points + center


def circle_hid_edge(xy, r, n_point=80):
    points = xy + np.array(
        [[r * m.cos(2 * ii * m.pi / n_point), r * m.sin(2 * ii * m.pi / n_point)] for ii in range(n_point)])
    plt.plot(points[:, 0], points[:, 1], color=(0, 0, 0), linewidth=1, linestyle='--')


def myarrow(ax, start_point, end_point, l=0.1, theta=25 / 180 * m.pi):
    theta_track = m.atan2((start_point - end_point)[1], (start_point - end_point)[0])
    point_l = end_point + l * np.array([m.cos(theta_track + theta), m.sin(theta_track + theta)])
    point_r = end_point + l * np.array([m.cos(theta_track - theta), m.sin(theta_track - theta)])
    ax.plot([end_point[0], point_l[0]], [end_point[1], point_l[1]], linewidth=1, color=(0, 0, 0), linestyle='-')
    ax.plot([end_point[0], point_r[0]], [end_point[1], point_r[1]], linewidth=1, color=(0, 0, 0), linestyle='-')


def plot_track(env, track, track_board):
    n_step = len(track)
    n_agents = track[0].shape[0]

    track_agent = [np.vstack([track[jj][ii] for jj in range(n_step)]) for ii in range(n_agents)]
    track_board = np.array(track_board)
    # size_ground = env.arena.size_ground
    size_ground = np.array([3.5, 5.5])

    # 图片设置
    fig, ax = plt.subplots()
    ax.set_xlim(-size_ground[0], size_ground[0])
    ax.set_ylim(-1, size_ground[1])
    ax.set_aspect('equal')

    # 遍历图形的四个边框，并分别设置它们的线宽
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # 画障碍物
    for jj in range(len(env.obstacle_pos)):
        if env.obstacle_arena.obs_pass[jj]:
            cir_color = (0, 1, 0)
        else:
            cir_color = (0.2, 0.3, 0.4)
        circle = Circle(xy=env.obstacle_pos[jj], radius=env.obstacle_size[jj, 0],
                        edgecolor=cir_color, facecolor=cir_color)  # 你可以改变facecolor来选择颜色
        ax.add_patch(circle)

    # 画起点
    for ii in range(n_agents):
        car_init_ii = np.array([env.l_form * m.cos(2 * m.pi / n_agents * ii + m.pi / 2),
                                env.l_form * m.sin(2 * m.pi / n_agents * ii + m.pi / 2)])
        circle = Circle(xy=car_init_ii, radius=0.1,  # env.robots[ii].robot.size[0]
                        edgecolor=(1, 0, 0), facecolor=(1, 0, 0))  # 你可以改变facecolor来选择颜色
        ax.add_patch(circle)

    circle = Circle(xy=(0.0, 0.0), radius=env.board_radius,
                    edgecolor=(0, 1, 1), facecolor=(0, 1, 1, 0.3))  # 你可以改变facecolor来选择颜色
    ax.add_patch(circle)

    # 画终点
    for ii in range(n_agents):
        target = env.target_board
        x = env.l_form * m.cos(2 * m.pi / n_agents * ii + m.pi / 2)
        y = env.l_form * m.sin(2 * m.pi / n_agents * ii + m.pi / 2)
        target_ii = target + np.array([x, y])

        star_points = create_star_points(0.1, target_ii)  # env.robots[ii].robot.size[0]
        star = Polygon(star_points, closed=True, edgecolor=(1, 0, 0), facecolor=(1, 0, 0))
        ax.add_patch(star)

    star_points = create_star_points(env.board_radius, env.target_board)
    star = Polygon(star_points, closed=True, edgecolor=(0, 1, 1), facecolor=(0, 1, 1, 0.3))
    ax.add_patch(star)

    # 画轨迹
    for ii in range(n_agents):
        ax.plot(track_agent[ii][:, 0], track_agent[ii][:, 1], linewidth=1.5, color=(0, 0, 0), linestyle='-')

    # 画轨迹(34, 139, 34)
    ax.plot(track_board[:, 0], track_board[:, 1], linewidth=1.5, color=(34 / 255, 139 / 255, 34 / 255), linestyle='-')

    # 画中间点
    for jj in np.linspace(0, n_step - 1, 4 + 1)[1:]:
        jj = int(jj)
        board_jj = track_board[jj, :]
        circle = Circle(xy=board_jj, radius=0.05,
                        edgecolor=(0, 0, 0), facecolor=(0, 0, 0, 1))
        ax.add_patch(circle)
        for ii in range(n_agents):
            robot_ii = track_agent[ii][jj, :]
            circle = Circle(xy=robot_ii, radius=0.1,  # env.robots[ii].robot.size[0]
                            edgecolor=(0, 0, 0), facecolor=(0, 0, 0, 0.3))
            ax.add_patch(circle)
            # circle_hid_edge(xy=robot_ii, r=env.robots[ii].robot.size[0])
            plt.plot([board_jj[0], robot_ii[0]], [board_jj[1], robot_ii[1]],
                     color=(0, 0, 0), linewidth=1, linestyle='--')

    plt.show()
