from robosuite.utils.input_utils import *
from stanford_su23.collect_vis_bcrnn_wrapper import BCRNN_DataCollectionWrapper
from robosuite.utils.transform_utils import euler2mat, mat2quat, quat_multiply
from stanford_su23.peg_hole.utils import get_env, linear_action, random_reset, encode_target_state, get_current_state, corrected_hole_pos
from robosuite.utils.binding_utils import MjSimState
import argparse
import numpy as np 
import math


def collect_demo(env, dilated):

    obs = random_reset(env)
    # while check_contact(env.sim, env.robots[0].robot_model) or check_contact(env.sim, env.robots[1].robot_model):
    env.render()

    # x, y, z -> out of screen, x-axis, y-axis
    env.record()

    robot1_target_pos = obs["robot1_eef_pos"]
    # robot1_target_quat = mat2quat(euler2mat(np.array([-np.pi, 0, 0])))
    robot1_target_quat = np.array([1, 0, 0, 0])
    robot0_target_pos = corrected_hole_pos(obs, dilated=dilated)
    robot0_target_pos[1] = robot1_target_pos[1] - 0.5
    # robot0_rot_quat = mat2quat(euler2mat(np.array([-np.pi/2, -np.pi, 0])))
    # robot0_target_quat = quat_multiply(robot0_rot_quat, robot1_target_quat)
    robot0_target_quat = np.array([0, math.sqrt(2)/2, math.sqrt(2)/2, 0])
    target_state = encode_target_state(robot0_target_pos, robot0_target_quat, robot1_target_pos, robot1_target_quat)
    def update_target_state(obs, target_state):
        ts = target_state.copy()
        ts[:3] = corrected_hole_pos(obs, dilated=dilated)
        ts[1] = ts[7] - 0.5
        return ts
    obs, _ = linear_action(env, target_state, update_target_state=update_target_state)
    target_state = get_current_state(obs)
    target_state[:3] = corrected_hole_pos(obs, dilated=dilated)
    target_state[1] -= 0.15
    obs, _ = linear_action(env, target_state)

    env.stop_record()
    env.flush()

def playback_demo(env, eps_file):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        eps_file (str): The path to the file containing data for an episode.
    """
    data = np.load(eps_file)

    env.reset()

    q_init = data["q"][0, :]
    qdot_init = data["qdot"][0, :]
    sim_state = MjSimState(0, q_init, qdot_init)
    env.sim.set_state(sim_state)
    env.sim.forward()
    # small error found when setting q directly so move to inital pose
    ee_position = data["ee_position"]
    ee_orientation = data["ee_orientation"]
    target_state = encode_target_state(ee_position[0,:][:3], ee_orientation[0,:][:4], ee_position[0,:][3:], ee_orientation[0, :][4:])
    linear_action(env, target_state, render=False)

    for pos, ori in zip(ee_position, ee_orientation):
        target_state = encode_target_state(pos[:3], ori[:4], pos[3:], ori[4:])
        linear_action(env, target_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="tmp/")
    parser.add_argument("--demos", type=int, default=1)
    parser.add_argument("--playback", type=str, default=None)
    parser.add_argument("--dilated", action="store_true")
    args = parser.parse_args()

    env = get_env(dilated=args.dilated)

    eps_file = args.playback
    if eps_file:
        playback_demo(env, eps_file)
    else:
        # wrap the environment with data collection wrapper
        env = BCRNN_DataCollectionWrapper(env, args.directory)
        for i in range(args.demos):
            collect_demo(env, args.dilated)
            print("finished demo {}".format(i))

    env.close()
    