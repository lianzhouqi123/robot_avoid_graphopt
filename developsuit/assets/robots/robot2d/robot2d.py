from dm_control import mjcf
import numpy as np
from developsuit.props.primitive import Primitive
from .axis_xy import Axis_xy


class Robot2d:
    def __init__(self):
        # 生成坐标轴
        self.axis = Axis_xy()
        self.mjcf_model = self.axis.base.mjcf_model
        self.joints = self.axis.joints
        self.joints_car = [self.axis.joint_x, self.axis.joint_y]

        self.size = np.array([0.1, 0.05])
        size_str = " ".join(str(x) for x in self.size)
        self.body = Primitive(name="obstacle", type="cylinder", size=size_str, rgba="1 0 0 1", conaffinity=1,
                              contype=1, option="geom", density="1000")
        framebody = self.axis.attach(self.body, np.array([0, 0, self.size[1]]))

    def get_pos(self, physics):
        return physics.bind(self.joints).qpos

    # geom
    @property
    def geoms(self):
        return [self.body.geom]

    # 轴geom
    @property
    def geoms_axis(self):
        return self.axis.geoms
