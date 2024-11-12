from dm_control import mjcf
import numpy as np
from developsuit.props.primitive import Primitive
from developsuit.assets.bodies.virtual_axis.axis_xyz import Axis_xyz


class Obstacle:
    def __init__(self, arena):
        self._arena = arena
        # 生成坐标轴
        self.axis = Axis_xyz(self._arena)
        self.joints = self.axis.joints

        self.body = None
        self.size = None

    def obstacle_create(self, size):
        size_str = " ".join(str(x) for x in size)
        self.body = Primitive(name="obstacle", type="cylinder", size=size_str, rgba="0 1 0 1", conaffinity=1,
                              contype=1, option="geom", density="1000")
        framebody = self.axis.attach(self.body)
        self.size = size

    def set_pos(self, physics, pos):
        pos = np.hstack((pos.reshape([-1]), np.array([self.size[-1]])))
        physics.bind(self.joints).qpos = pos

    def get_pos(self, physics):
        return physics.bind(self.joints).qpos

    # 障碍物geom
    @property
    def geom(self):
        return [self.body.geom]

    # 障碍物轴geom
    @property
    def geom_axis(self):
        return self.axis.geom
