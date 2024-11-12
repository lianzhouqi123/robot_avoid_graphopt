from dm_control import mjcf
import numpy as np
from developsuit.props.primitive import Primitive


class Axis_xy:
    def __init__(self, arena):
        self._arena = arena
        self.axis_x = Primitive(name="axis_x", type="sphere", size="0.01", rgba="0 0 0 0", conaffinity=0,
                                contype=0, option="geom")
        self.axis_y = Primitive(name="axis_y", type="sphere", size="0.01", rgba="0 0 0 0", conaffinity=0,
                                contype=0, option="geom")

        frame_axis_x = self._arena.attach(self.axis_x.mjcf_model)
        joint_x = frame_axis_x.add('joint', type='slide', axis=[1, 0, 0])
        frame_axis_y = self.axis_x.mjcf_model.attach(self.axis_y.mjcf_model)
        joint_y = frame_axis_y.add('joint', type='slide', axis=[0, 1, 0])
        self.joints = [joint_x, joint_y]

    def attach(self, child):
        frame_child = self.axis_y.mjcf_model.attach(child.mjcf_model)
        return frame_child

    @property
    def geom(self):
        return [self.axis_x.geom, self.axis_y.geom]
