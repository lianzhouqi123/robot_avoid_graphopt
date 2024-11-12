from dm_control import mjcf
import numpy as np
from developsuit.props.primitive import Primitive


class Board:
    def __init__(self, arena, board_size):
        # # generate x and y axis
        self._arena = arena
        board_size_str = " ".join(str(x) for x in board_size)
        # generate obstacle
        self.board = Primitive(name="board", type="box", size=board_size_str, rgba="0 0.6 0.6 1", conaffinity=0,
                               contype=0, option="geom", friction="1.0 0 0", density="50")

        self.frame_board = self._arena.attach(self.board.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0])
        self.joint_board = self.frame_board.add("joint", type="free")
        self.joints = [self.joint_board]

        self.geom = self.board.geom
