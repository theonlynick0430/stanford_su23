import sys
from robosuite.utils.input_utils import *
from stanford_su23.collect_vis_bcrnn_wrapper import BCRNN_DataCollectionWrapper
from robosuite.utils.transform_utils import euler2mat, mat2quat, quat_multiply
from stanford_su23.peg_hole.utils import get_env, linear_action, random_reset, encode_target_state, get_current_state, check_contact
from robosuite.utils.binding_utils import MjSimState
import argparse
import numpy as np 
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


def collect_demo(env, sqr_radius):

    obs = random_reset(env)
    # while check_contact(env.sim, env.robots[0].robot_model) or check_contact(env.sim, env.robots[1].robot_model):
    env.render()

    # x, y, z -> out of screen, x-axis, y-axis
    env.record()
    sqr_target_pos = np.array([0, 0.25, 1.45])
    sqr_target_pos = np.random.normal(sqr_target_pos, 1e-2)
    sqr_target_quat = mat2quat(euler2mat(np.array([-np.pi, 0, 0])))
    peg_target_pos = sqr_target_pos.copy()
    peg_target_pos[1] = -peg_target_pos[1]
    peg_target_pos[2] -= sqr_radius
    peg_rot_quat = mat2quat(euler2mat(np.array([-np.pi/2, -np.pi, 0])))
    peg_target_quat = quat_multiply(peg_rot_quat, sqr_target_quat)
    target_state = encode_target_state(peg_target_pos, peg_target_quat, sqr_target_pos, sqr_target_quat)
    obs, _ = linear_action(env, target_state)
    peg_to_hole = obs["peg_to_hole"]
    target_state = get_current_state(obs)
    adjust = np.array([-0.1, 0, 0])
    target_state[:3] += peg_to_hole+adjust
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
    
    sqr_radius = 0.17
    if args.dilated:
        sqr_radius = 0.085

    env = get_env(dilated=args.dilated)

    eps_file = args.playback
    if eps_file:
        playback_demo(env, eps_file)
    else:
        # wrap the environment with data collection wrapper
        env = BCRNN_DataCollectionWrapper(env, args.directory)
        for i in range(args.demos):
            collect_demo(env, sqr_radius)
            print("finished demo {}".format(i))

    env.close()
    