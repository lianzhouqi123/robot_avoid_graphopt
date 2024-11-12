import numpy as np
from developsuit.utils.rot import *
from developsuit.utils.transform_utils import *
import math as m


def fkine(DH, q, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1])):
    # 标准DH参数
    q = q.reshape([-1, 1])
    DH = np.copy(DH)
    Ndof = q.size
    if DH_mode is None:
        DH_mode = ["hinge"] * Ndof

    temp = np.eye(4)
    temp[0:3, 0:3] = R_base  # 基座旋转矩阵
    temp[0:3, 3] = base.reshape([3])  # 基座向量
    T = np.zeros([4, 4, Ndof])  # 初始化T

    # DH第一列为theta的offset, 在这上面加关节角
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            DH[ii, 0] += q[ii, 0]
        if DH_mode[ii] == "slide":
            DH[ii, 1] += q[ii, 0]

    for ii in range(Ndof):
        ct = m.cos(DH[ii, 0])
        st = m.sin(DH[ii, 0])
        ca = m.cos(DH[ii, 3])
        sa = m.sin(DH[ii, 3])

        temp = temp @ np.array([[ct, -st * ca, st * sa, DH[ii, 2] * ct],
                                [st, ct * ca, -ct * sa, DH[ii, 2] * st],
                                [0, sa, ca, DH[ii, 1]],
                                [0, 0, 0, 1]])

        T[:, :, ii] = temp

    return T


def jac(DH, q, dq=None, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1])):
    # https://zhuanlan.zhihu.com/p/205342861?utm_id=0
    q = q.reshape([-1])
    DH = np.copy(DH)
    Ndof = DH.shape[0]
    if DH_mode is None:
        DH_mode = ["hinge"] * Ndof

    if dq is None:
        dq = np.zeros([Ndof, 1])
    dq = dq.reshape([Ndof, 1])

    T = fkine(DH, q, DH_mode=DH_mode, R_base=R_base, base=base)  # 所有的T
    R = T[0:3, 0:3, :]  # 各系旋转矩阵
    x = T[0:3, 3, :].reshape([3, -1])  # 各系的原点位置
    x_end = x[:, -1].reshape([-1, 1])  # 末端位置

    omega_save = np.zeros([3, Ndof])

    R_old = R_base
    omega_old = omega_base
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            omega = omega_old + R_old @ np.array([0, 0, dq[ii, 0]]).reshape([-1, 1])  # 各坐标系的角速度
        else:
            omega = omega_old

        # 保存参数
        omega_save[:, ii] = omega.reshape([3])  # 对应于 q(ii+1)

        # 更新参数
        R_old = R[:, :, ii]
        omega_old = omega

    # 对应于网页式(8)
    dx = np.zeros([3, 1])  # end 相对于 6(=end) 的速度
    dX = np.zeros([3, Ndof])  # 0为end相对base，Ndof-1为end相对于5, 即与关节角对应
    for ii in range(Ndof - 1, 0, -1):  # Ndof-1 : -1 : 1
        if DH_mode[ii] == "hinge":
            dx += np.cross(omega_save[:, ii].reshape([1, 3]), (x[:, ii] - x[:, ii - 1]).reshape([1, 3])).reshape([3, 1])
        else:
            dx += R[:, :, ii - 1] @ np.array([0, 0, dq[ii, 0]]).reshape([3, 1])
        dX[:, ii] = dx.reshape([3])

    dX[:, 0] = dx.reshape([3]) + np.cross(omega_save[:, 0].reshape([3]),
                                          (x[:, 0].reshape([3]) - base.reshape([3]))).reshape([3])

    # qi 在 i-1系下描述
    Z = np.zeros([3, Ndof])  # Z轴矢量
    U = np.zeros([3, Ndof])  # Z矢量叉乘（x_end - xi）
    dZ = np.zeros([3, Ndof])
    dU = np.zeros([3, Ndof])

    # 对应于网页式(3)
    Z[:, 0] = (R_base @ np.array([0, 0, 1]).reshape([3, 1])).reshape([3])
    # 对应于网页式(3)
    U[:, 0] = np.cross(Z[:, 0].reshape([3]), (x_end.reshape([3]) - base.reshape([3]))).reshape([3])
    # 对应于网页式(10)
    dZ[:, 0] = np.cross(omega_base.reshape([3]), Z[:, 0].reshape([3])).reshape([3])
    # 对应于网页式(7)
    dU[:, 0] = (((np.cross(Z[:, 0].reshape([3]), dX[:, 0].reshape([3]))
                  + np.cross(dZ[:, 0].reshape([3]), x_end.reshape([3]) - base.reshape([3])))).reshape([3]))

    for ii in range(Ndof - 1):
        # 对应于网页式(3)
        Z[:, ii + 1] = (R[:, :, ii] @ np.array([0, 0, 1]).reshape([3, 1])).reshape([3])
        # 对应于网页式(3)
        U[:, ii + 1] = np.cross(Z[:, ii + 1].reshape([3]), (x_end.reshape([3]) - x[:, ii].reshape([3]))).reshape([3])
        # 对应于网页式(10)
        dZ[:, ii + 1] = np.cross(omega_save[:, ii].reshape([3]), Z[:, ii + 1].reshape([3])).reshape([3])
        # 对应于网页式(7)
        dU[:, ii + 1] = (((np.cross(Z[:, ii + 1].reshape([3]), dX[:, ii + 1].reshape([3]))
                           + np.cross(dZ[:, ii + 1].reshape([3]), x_end.reshape([3]) - x[:, ii].reshape([3]))))
                         .reshape([3]))

    J = np.zeros([6, Ndof])
    dJ = np.zeros([6, Ndof])
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            J[0:3, ii] = U[:, ii]  # 对应于网页式(4)(5)
            J[3:6, ii] = Z[:, ii]
            dJ[0:3, ii] = dU[:, ii]  # 对应于网页式(6)
            dJ[3:6, ii] = dZ[:, ii]
        elif DH_mode[ii] == "slide":
            J[0:3, ii] = Z[:, ii]
            dJ[0:3, ii] = dZ[:, ii]

    return J, dJ


def fkine_ee(DH, q, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1])):
    T = fkine(DH, q, DH_mode=DH_mode, R_base=R_base, base=base)
    x_end = T[0:3, 3, -1].reshape([3, 1])
    R_end = T[0:3, 0:3, -1]
    quat_end = mat2quat(R_end).reshape([4, 1])

    return x_end, quat_end


def ikine(DH, qc, xd, quatd, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1]),
          threshold=1e-5, max_iter=5e2):
    # quat cos位在第一位
    DH = np.copy(DH)
    qc = qc.reshape([-1, 1])
    xd = xd.reshape([-1, 1])
    quatd = quatd.reshape([4, 1])
    x, quat = fkine_ee(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base)

    euld = quat2eul(quatd).reshape([3, 1])

    epsino = m.pi / 2 - 5e-1
    # 判断欧拉角模式，避开奇异点
    if abs(euld[1, 0]) < epsino:
        mode = "ZYX"
    else:
        mode = "XZY"

    eul = quat2eul(quat, mode).reshape([3, 1])
    euld = quat2eul(quatd, mode).reshape([3, 1])

    x = np.vstack((x, eul))
    xd = np.vstack((xd, euld))
    J, _ = jac(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)

    err_p = np.linalg.norm(x[0:3] - xd[0:3])
    err_eul = np.linalg.norm(x[3:6] - xd[3:6])
    err = err_p + err_eul

    ii = 0
    while 1:
        ii += 1
        delta_x = xd - x
        pinvJ = np.linalg.pinv(J)
        delta_q = pinvJ @ delta_x
        # if err < 1:
        #     qc = qc + 0.06 * err * delta_q
        # else:
        qc = qc + 0.1 * delta_q
        qc = clip_q(qc)
        x, quat = fkine_ee(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base)
        eul = quat2eul(quat, mode).reshape([3, 1])
        x = np.vstack((x, eul))
        J, _ = jac(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)

        err_p = np.linalg.norm(x[0:3] - xd[0:3])
        err_eul = np.linalg.norm(x[3:6] - xd[3:6])
        err = err_p + err_eul

        if err < threshold:
            break

        if ii > max_iter:
            print('**ikine** breaks after %d iterations with error %.4f.\n'.format(ii, err))
            break

    return qc, err


def ikine_3d(DH, qc, xd, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1]),
             threshold=1e-6, max_iter=5e2):
    DH = np.copy(DH)
    qc = qc.reshape([-1, 1])
    xd = xd.reshape([-1, 1])
    x, quat = fkine_ee(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base)
    x = x.reshape([-1, 1])
    J, _ = jac(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)
    J = J[0:3, :]  # 只要位置

    ii = 0
    while 1:
        ii += 1
        delta_x = xd - x
        pinvJ = np.linalg.pinv(J)
        delta_q = pinvJ @ delta_x
        qc = qc + 0.1 * delta_q
        qc = clip_q(qc)
        x, quat = fkine_ee(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base)
        x = x.reshape([-1, 1])
        J, _ = jac(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)
        J = J[0:3, :]  # 只要位置
        err = np.linalg.norm(x[0:3, 0] - xd[0:3, 0])

        if err < threshold:
            break

        if ii > max_iter:
            print('**ikine** breaks after %d iterations with error %.4f.\n'.format(ii, err))
            break

    return qc, err


# def forward_kine(joint):
#     """
#     Purpose: forward kinematics of the RM63 robot
#     input: joint, shape=(6,), dim=1, joint_list 1~6
#     """
#     R1 = np.dot(rotz(180), rotz(joint[0]))
#     R2 = np.dot(np.dot(rotz(90), roty(-90)), rotz(joint[1]))
#     R3 = rotz(joint[2])
#     R4 = np.dot(np.dot(roty(90), rotz(90)), rotz(joint[3]))
#     R5 = np.dot(rotx(-90), rotz(joint[4]))
#     R6 = np.dot(rotx(90), rotz(joint[5]))
#     R7 = rotz(30.69)
#
#     p1 = np.array([0, 0, 0.108])
#     p2 = np.array([0.086, 0, 0.064])
#     p3 = np.array([0.380, 0, 0])
#     p4 = np.array([0.083, 0.086, 0])
#     p5 = np.array([0, 0, 0.322])
#     p6 = np.array([0, -0.1435, 0])
#     p7 = np.array([0, 0, 0.210])
#
#     T1 = np.hstack((R1, p1.reshape(-1, 1)))
#     T2 = np.hstack((R2, p2.reshape(-1, 1)))
#     T3 = np.hstack((R3, p3.reshape(-1, 1)))
#     T4 = np.hstack((R4, p4.reshape(-1, 1)))
#     T5 = np.hstack((R5, p5.reshape(-1, 1)))
#     T6 = np.hstack((R6, p6.reshape(-1, 1)))
#     T7 = np.hstack((R7, p7.reshape(-1, 1)))
#
#     T1 = np.vstack((T1, np.array([0, 0, 0, 1])))
#     T2 = np.vstack((T2, np.array([0, 0, 0, 1])))
#     T3 = np.vstack((T3, np.array([0, 0, 0, 1])))
#     T4 = np.vstack((T4, np.array([0, 0, 0, 1])))
#     T5 = np.vstack((T5, np.array([0, 0, 0, 1])))
#     T6 = np.vstack((T6, np.array([0, 0, 0, 1])))
#     T7 = np.vstack((T7, np.array([0, 0, 0, 1])))
#
#     D1 = T1
#     D2 = np.dot(D1, T2)
#     D3 = np.dot(D2, T3)
#     D4 = np.dot(D3, T4)
#     D5 = np.dot(D4, T5)
#     D6 = np.dot(D5, T6)
#     D7 = np.dot(D6, T7)
#     np.concatenate((D1[np.newaxis, :], D2[np.newaxis, :], D3[np.newaxis, :], D4[np.newaxis, :], D5[np.newaxis, :],
#                     D6[np.newaxis, :]))
#
#     return D1, D2, D3, D4, D5, D6, D7


# def compute_each_joint_sapcepos(joint):
#     # joint = np.degrees(joint)
#     R1 = np.dot(rotz(180), rotz(joint[0]))
#     R2 = np.dot(np.dot(rotz(90), roty(-90)), rotz(joint[1]))
#     R3 = rotz(joint[2])
#     R4 = np.dot(np.dot(roty(90), rotz(90)), rotz(joint[3]))
#     R5 = np.dot(rotx(-90), rotz(joint[4]))
#     R6 = np.dot(rotx(90), rotz(joint[5]))
#     R7 = np.eye(3)
#     R8 = np.eye(3)
#     R9 = np.eye(3)
#
#     p1 = np.array([0, 0, 0.108])
#     p2 = np.array([0.086, 0, 0.064])
#     p3 = np.array([0.380, 0, 0])
#     p4 = np.array([0.083, 0.086, 0])
#     p5 = np.array([0, 0, 0.322])
#     p6 = np.array([0, -0.1435, 0])
#     p7 = np.array([0, 0, 0.095])
#     p8 = np.array([0.0725, 0, 0.210])
#     p9 = np.array([-0.0725, 0, 0.210])
#
#     T1 = np.hstack((R1, p1.reshape(-1, 1)))
#     T2 = np.hstack((R2, p2.reshape(-1, 1)))
#     T3 = np.hstack((R3, p3.reshape(-1, 1)))
#     T4 = np.hstack((R4, p4.reshape(-1, 1)))
#     T5 = np.hstack((R5, p5.reshape(-1, 1)))
#     T6 = np.hstack((R6, p6.reshape(-1, 1)))
#     T7 = np.hstack((R7, p7.reshape(-1, 1)))
#     T8 = np.hstack((R8, p8.reshape(-1, 1)))
#     T9 = np.hstack((R9, p9.reshape(-1, 1)))
#
#     T1 = np.vstack((T1, np.array([0, 0, 0, 1])))
#     T2 = np.vstack((T2, np.array([0, 0, 0, 1])))
#     T3 = np.vstack((T3, np.array([0, 0, 0, 1])))
#     T4 = np.vstack((T4, np.array([0, 0, 0, 1])))
#     T5 = np.vstack((T5, np.array([0, 0, 0, 1])))
#     T6 = np.vstack((T6, np.array([0, 0, 0, 1])))
#     T7 = np.vstack((T7, np.array([0, 0, 0, 1])))
#     T8 = np.vstack((T8, np.array([0, 0, 0, 1])))
#     T9 = np.vstack((T9, np.array([0, 0, 0, 1])))
#
#     # compute the position of each joint
#     D1 = T1
#     D2 = np.dot(T1, T2)
#     D3 = np.dot(D2, T3)
#     D4 = np.dot(D3, T4)
#     D5 = np.dot(D4, T5)
#     D6 = np.dot(D5, T6)
#     D7 = np.dot(D6, T7)
#     D8 = np.dot(D6, T8)
#     D9 = np.dot(D6, T9)
#
#     p_q1 = D1[0:3, 3].reshape(1, -1)
#     p_q2 = D2[0:3, 3].reshape(1, -1)
#     p_q3 = D3[0:3, 3].reshape(1, -1)
#     p_q4 = D4[0:3, 3].reshape(1, -1)
#     p_q5 = D5[0:3, 3].reshape(1, -1)
#     p_q6 = D6[0:3, 3].reshape(1, -1)
#     p_q7 = D7[0:3, 3].reshape(1, -1)
#     p_q8 = D8[0:3, 3].reshape(1, -1)
#     p_q9 = D9[0:3, 3].reshape(1, -1)
#     return p_q1, p_q2, p_q3, p_q4, p_q5, p_q6, p_q7, p_q8, p_q9


def arm_obstacle_distance_detection(joint, obstacle_pos, obstacle_radius, seg_num: list):
    """
    Purpose: 
        compute the distance between the arm and the obstacle

    Args:
        joint: each joint angle of the arm
        obstacle_pos: the position of the obstacle
        obstacle_radius: the radius of the obstacle
        seg_num: the number of the segment points of "each" link

    Returns:
        flag: if distance is less than the range of obstacle discriminant ball, flag is True, else False;

        distance: the distance among the obstacle and detection points of the arm;
    """
    p_base = np.array([0, 0, 0])
    p_q1, p_q2, p_q3, p_q4, p_q5, p_q6, p_q7, p_q8, p_q9 = compute_each_joint_sapcepos(joint)
    link1_point = spaceline_segment(p_base, p_q1, seg_num[0])
    link2_point = spaceline_segment(p_q1, p_q2, seg_num[1])
    link3_point = spaceline_segment(p_q2, p_q3, seg_num[2])
    link4_point = spaceline_segment(p_q3, p_q4, seg_num[3])
    link5_point = spaceline_segment(p_q4, p_q5, seg_num[4])
    link6_point = spaceline_segment(p_q5, p_q6, seg_num[5])
    link7_point = spaceline_segment(p_q6, p_q7, seg_num[6])
    link8_point = spaceline_segment(p_q7, p_q8, seg_num[7])
    link9_point = spaceline_segment(p_q7, p_q9, seg_num[8])
    link_point = np.vstack((p_base, link1_point, p_q1,
                            link2_point, p_q2,
                            link3_point, p_q3,
                            link4_point, p_q4,
                            link5_point, p_q5,
                            link6_point, p_q6,
                            link7_point, p_q7,
                            link8_point, p_q8,
                            link9_point, p_q9))
    distance = np.zeros((1, link_point.shape[0]))
    for i in range(link_point.shape[0]):
        distance[0, i] = np.linalg.norm(link_point[i, :] - obstacle_pos)

    # test the distance between the obstacle and the arm
    if np.any(distance <= obstacle_radius * 2):
        # 返回值为"True"，则为碰撞
        flag = True
    else:
        # 返回值为"False"，则为非碰撞
        flag = False

    return flag, distance


def spaceline_segment(base_point: np.array, destination_point: np.array, seg_num: int) -> np.array:
    """
    Purpose: compute the segment point of the space line
    input: base_point, destination_point, seg_num
    return: segment_point (n, 3), n is the number of segment points in column(up to down)
                        [[x1, y1, z1],
                         [x2, y2, z2],
                         ...,
                         [xn, yn, zn]]
    base_point: the start point of the space line, unit: m
    destination_point: the end point of the space line, unit: m
    """
    destination_point = np.array(destination_point).reshape(1, -1)
    base_point = np.array(base_point).reshape(1, -1)
    x_part = np.zeros((1, seg_num))
    y_part = np.zeros((1, seg_num))
    z_part = np.zeros((1, seg_num))
    x_i = (destination_point[0, 0] - base_point[0, 0]) / (seg_num + 1)
    y_i = (destination_point[0, 1] - base_point[0, 1]) / (seg_num + 1)
    z_i = (destination_point[0, 2] - base_point[0, 2]) / (seg_num + 1)
    for i in range(seg_num):
        x_part[0, i] = base_point[0, 0] + i * x_i
        y_part[0, i] = base_point[0, 1] + i * y_i
        z_part[0, i] = base_point[0, 2] + i * z_i

    x_part = x_part.reshape(-1, 1)
    y_part = y_part.reshape(-1, 1)
    z_part = z_part.reshape(-1, 1)
    segment_point = np.hstack((x_part, y_part, z_part))

    return segment_point
