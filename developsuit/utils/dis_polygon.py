import numpy as np


def mycross(a, b):
    a = a.reshape([-1, 2])
    b = b.reshape([2])
    c = a[:, 0] * b[1] - a[:, 1] * b[0]
    return c


def dot_seq(a, b):
    a = a.reshape([-1, 2])
    b = b.reshape([-1, 2])
    c = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    return c


def is_point_in_polygon(polygon, points):
    """
    使用射线法检查点是否在多边形内部
    :param points: 一个形状为[n, 2]的numpy数组，表示n个点的坐标
    :param polygon: 一个形状为[n, 2]的numpy数组，表示多边形的顶点坐标
    :return: 如果点在多边形内部，则返回True；否则返回False
    """
    n_points = points.shape[0]
    x = points[:, 0]
    y = points[:, 1]
    n = len(polygon)
    inside = np.array([False] * n_points)

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if p2y - p1y > 0:
            # 向上的边包含起点
            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            # p1y <= y < p2y and x <= xinters
            logic1 = np.logical_and(np.logical_and(p1y <= y, y < p2y), x <= xinters)
            inside[logic1] = np.logical_not(inside[logic1])
        elif p2y - p1y < 0:
            # 向下的边包含终点
            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            # p2y <= y < p1y and x <= xinters
            logic2 = np.logical_and(np.logical_and(p2y <= y, y < p1y), x <= xinters)
            inside[logic2] = np.logical_not(inside[logic2])

        p1x, p1y = p2x, p2y

    return inside


def dis_polygon(polygon, points):
    """
    计算点到多边形边界的距离
    :param polygon: 一个形状为[n, 2]的numpy数组，表示多边形的顶点坐标
    :param points: 一个形状为[n, 2]的numpy数组，表示n个点的坐标
    :return: 点到多边形边界的距离（如果点在多边形内部，则返回负数或根据需求调整）
    """
    n_points = points.shape[0]
    min_distance = np.ones([n_points]) * 1e6
    inside = is_point_in_polygon(polygon, points)
    min_distance[inside] = -1  # 假设在内部时返回-1
    outside = np.logical_not(inside)

    for i in range(len(polygon)):
        v0 = polygon[i]
        v1 = polygon[(i + 1) % len(polygon)]
        # 计算v0到v1的向量
        vec = v1 - v0  # [2]
        # 计算法线向量（垂直于v0v1）
        normal = np.array([-vec[1], vec[0]])  # [2]
        # 标准化法线向量
        normal /= np.linalg.norm(normal)  # [2]
        # 计算点到直线的距离公式（使用法线向量）
        distance_to_line = np.abs(np.dot(normal, (points - v0).T))  # [n]
        chuizu = points - distance_to_line.reshape([n_points, 1]) * normal.reshape([1, 2])  # [n, 2]
        # 修正垂足
        # not np.abs(np.cross(chuizu - v0, v0 - v1)) < 1e-4
        chuizu[np.logical_not(np.abs(mycross(chuizu - v0, v0 - v1)) < 1e-4)] = \
            (points + distance_to_line.reshape([n_points, 1]) * normal.reshape([1, 2]))[
                np.logical_not(np.abs(mycross(chuizu - v0, v0 - v1)) < 1e-4)]

        # 垂足在外，则最小距离为端点
        logic1 = np.logical_and(dot_seq(chuizu - v0, chuizu - v1) > 0,
                                        np.linalg.norm(chuizu - v0, axis=1) > np.linalg.norm(chuizu - v1, axis=1))
        distance_to_line[logic1]  = np.linalg.norm(points - v1, axis=1)[logic1]
        logic2 = np.logical_and(dot_seq(chuizu - v0, chuizu - v1) > 0,
                                np.linalg.norm(chuizu - v0, axis=1) < np.linalg.norm(chuizu - v1, axis=1))
        distance_to_line[logic2] = np.linalg.norm(points - v0, axis=1)[logic2]

        min_distance[outside] = np.minimum(min_distance[outside], distance_to_line[outside])

    return min_distance
