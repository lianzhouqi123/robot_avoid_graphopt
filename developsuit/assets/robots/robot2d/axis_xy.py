from dm_control import mjcf
import numpy as np
from developsuit.props.primitive import Primitive


class Axis_xy:
    def __init__(self):
        self.base = Primitive(name="base", type="sphere", size="0.01", rgba="0 0 0 0", conaffinity=0,
                              contype=0, option="geom")
        self.axis_x = Primitive(name="axis_x", type="sphere", size="0.01", rgba="0 0 0 0", conaffinity=0,
                                contype=0, option="geom")
        self.axis_y = Primitive(name="axis_y", type="sphere", size="0.01", rgba="0 0 0 0", conaffinity=0,
                                contype=0, option="geom")

        frame_axis_x = self.base.mjcf_model.attach(self.axis_x.mjcf_model)
        self.joint_x = frame_axis_x.add('joint', type='slide', axis=[1, 0, 0])
        frame_axis_y = self.axis_x.mjcf_model.attach(self.axis_y.mjcf_model)
        self.joint_y = frame_axis_y.add('joint', type='slide', axis=[0, 1, 0])
        self.joints = [self.joint_x, self.joint_y]

    def attach(self, child, pos=None):
        frame_child = self.axis_y.mjcf_model.attach(child.mjcf_model)
        if pos is not None:
            frame_child.pos = pos
        return frame_child

    @property
    def geoms(self):
        return [self.base, self.axis_x.geom, self.axis_y.geom]
