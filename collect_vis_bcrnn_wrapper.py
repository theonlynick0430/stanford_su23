"""
This data collection wrapper is useful for collecting demonstrations for BC-RNN.
"""

import os
import time
import numpy as np
from robosuite.wrappers import Wrapper
from stanford_su23.peg_hole.utils import corrected_hole_pos


class BCRNN_DataCollectionWrapper(Wrapper):

    def __init__(self, env, directory):
        """
        Initializes the data collection wrapper for BC-RNN.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
        """
        super().__init__(env)

        self.has_interaction = False
        self.directory = directory
        self.demo = None
        self.obs = None
        self._record = False

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

    def _reset_data(self):
        # clear data to be saved on disk  
        self.q = np.empty((0, len(self.sim.data.qpos)), dtype=np.float64)
        self.qdot = np.empty((0, len(self.sim.data.qvel)), dtype=np.float64)
        self.ee_position = np.empty((0, 6), dtype=np.float64)
        self.ee_orientation = np.empty((0, 8), dtype=np.float64)
        # birdview img
        idx = self.camera_names.index("birdview")
        h = self.camera_heights[idx]
        w = self.camera_widths[idx]
        self.image = np.empty((0, h, w, 3), dtype=np.uint8)
        # agentview img
        idx = self.camera_names.index("agentview")
        h = self.camera_heights[idx]
        w = self.camera_widths[idx]
        self.ego_image = np.empty((0, h, w, 3), dtype=np.uint8)
        self.action = np.empty((0, self.action_dim), dtype=np.float64)    

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).
        """
        self.has_interaction = True

        if self.demo is not None:
            self.demo += 1
        else:
            self.demo = 0

        self._reset_data()

    def record(self):
        self._record = True

    def stop_record(self):
        self._record = False

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection
        Must call flush before reset or data will be lost
        
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        obs = super().reset()
        self.has_interaction = False
        self.obs = obs
        return obs

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take iaaÂ~~~~~~~~~~~`` Zn environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        if self._record:
            # on the first time step, make directories for logging
            if not self.has_interaction:
                self._on_first_interaction()

            # save data 
            state = self.sim.get_state()
            self.q = np.append(self.q, [state.qpos], axis=0)
            self.qdot = np.append(self.qdot, [state.qvel], axis=0)
            # robot1_pos = self.obs["robot1_eef_pos"]
            hole_pos = corrected_hole_pos(self.obs)
            robot1_quat = self.obs["robot1_eef_quat"]
            # robot0_pos = self.obs["robot0_eef_pos"]
            peg_pos = self.obs["hole_pos"]-self.obs["peg_to_hole"]
            robot0_quat = self.obs["robot0_eef_quat"]
            # self.ee_position = np.append(self.ee_position, [np.concatenate((robot0_pos, robot1_pos))], axis=0)
            self.ee_position = np.append(self.ee_position, [np.concatenate((peg_pos, hole_pos))], axis=0)
            self.ee_orientation = np.append(self.ee_orientation, [np.concatenate((robot0_quat, robot1_quat))], axis=0)
            self.image = np.append(self.image, [self.obs["birdview_image"]], axis=0)
            self.ego_image = np.append(self.ego_image, [self.obs["agentview_image"]], axis=0)
            self.action = np.append(self.action, [action], axis=0)

        self.obs, reward, complete, misc = super().step(action)

        return self.obs, reward, complete, misc

    def flush(self):
        """
        Method to flush internal state to disk after episode has ended.
        It is the user's responsibilty to save data to disk before ending programs.
        """
        eps_path = os.path.join(self.directory, "demo_{}.npz".format(self.demo))
        np.savez(
            eps_path, 
            q=self.q,
            qdot=self.qdot,
            ee_position=self.ee_position,
            ee_orientation=self.ee_orientation,
            image=self.image,
            ego_image=self.ego_image,
            action=self.action
        )

