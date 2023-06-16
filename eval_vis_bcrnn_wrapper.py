"""
This data collection wrapper is useful for evaluating BC-RNN models.
"""

from attrdict import AttrDict as d
from robosuite.wrappers import Wrapper
import numpy as np
from stanford_su23.peg_hole.utils import corrected_hole_pos


class BCRNN_ModelEvaluationWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.obs = None

    def get_input(self):
        # robot0_pos = self.obs["robot0_eef_pos"]
        robot0_quat = self.obs["robot0_eef_quat"]
        # robot1_pos = self.obs["robot1_eef_pos"]
        robot1_quat = self.obs["robot1_eef_quat"]
        input = d()
        hole_pos = corrected_hole_pos(self.obs, dilated=True)
        peg_pos = self.obs["hole_pos"]-self.obs["peg_to_hole"]
        input["ee_position"] = np.array([np.concatenate((peg_pos, hole_pos))])
        # input["ee_position"] = np.array([np.concatenate((robot0_pos, robot1_pos))])
        input["ee_orientation"] = np.array([np.concatenate((robot0_quat, robot1_quat))])
        input["image"] = np.array([self.obs["birdview_image"]])
        input["ego_image"] = np.array([self.obs["agentview_image"]])
        return input

    def reset(self):
        self.obs = super().reset()
        return self.obs

    def step(self, action):
        self.obs, a, b, c = super().step(action)
        return self.obs, a, b, c, 