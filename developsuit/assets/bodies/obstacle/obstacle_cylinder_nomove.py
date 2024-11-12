from dm_control import mjcf
import numpy as np
from developsuit.props.primitive import Primitive
from developsuit.assets.bodies.virtual_axis.axis_xyz import Axis_xyz


class Obstacle:
    def __init__(self, arena):
        self._arena = arena

        self.body = None
        self.framebody = None
        self.size = None

    def obstacle_create(self, size, obs_pass=True):
        if obs_pass:
            color = "0 1 0 1"
        else:
            color = "0.2 0.3 0.4 1"
        size_str = " ".join(str(x) for x in size)
        self.body = Primitive(name="obstacle", type="cylinder", size=size_str, rgba=color, conaffinity=1,
                              contype=1, option="geom", density="1000")
        self.framebody = self._arena.attach(self.body.mjcf_model)
        self.size = size

    def set_pos(self, pos):
        pos = np.hstack((pos.reshape([-1]), np.array([self.size[-1]])))
        self.framebody.pos = pos

    # 障碍物geom
    @property
    def geom(self):
        return [self.body.geom]
