import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../.."))

from stanford_su23.latent_stabilize.utils import get_env, get_current_state, linear_action, single_action, gripper_action, inverse_quaternion
from robosuite.utils.transform_utils import quat_multiply, quat2axisangle, quat2mat, mat2euler
import numpy as np
from scipy.spatial.transform import Rotation
import math
import time

RENDER = False

def eul_delta(quat, target_quat):
    # returns target - current
    quat_delta = quat_multiply(target_quat, inverse_quaternion(quat))
    eul_delta = mat2euler(quat2mat(quat_delta)) # safe because delta is so small that we don't get gymbal lock
    return eul_delta

def init_pot(env):
    # workspace for left/right arm:
    # min: [-0.25, -0.25/0, 1]
    # max: [0.25, 0/0.25, 1.5]

    # find width of pot
    obs = env._get_observations(force_update=True)
    pot_width = np.linalg.norm(obs["handle0_xpos"]-obs["handle1_xpos"])

    # grasp handles
    obs = gripper_action(env, grasp=True, render=RENDER)
    rot_n90 = np.array([0, 0, -math.sqrt(2)*0.5, math.sqrt(2)*0.5])
    target_state = get_current_state(obs)
    target_state[0] += obs["gripper0_to_handle0"]
    target_state[0][2] += 0.1
    target_state[1] = quat_multiply(rot_n90, quat_multiply(obs["pot_quat"], target_state[1]))
    target_state[2] += obs["gripper1_to_handle1"]
    target_state[2][2] += 0.1
    target_state[3] = quat_multiply(rot_n90, quat_multiply(obs["pot_quat"], target_state[3]))
    obs, success = linear_action(env, target_state, max_steps=1000, render=RENDER)
    obs = gripper_action(env, grasp=False, render=RENDER)
    target_state = get_current_state(obs)
    target_state[0] += obs["gripper0_to_handle0"]
    target_state[0][2] -= 0.05
    # target_state[0][2] -= 0.075
    target_state[2] += obs["gripper1_to_handle1"]
    target_state[2][2] -= 0.05
    # target_state[2][2] -= 0.075
    obs, success = linear_action(env, target_state, render=RENDER)
    obs = gripper_action(env, grasp=True, render=RENDER)

    # pick up
    pot_init_height = 0.25
    target_state = get_current_state(obs)
    target_state[0][2] += pot_init_height
    target_state[2][2] += pot_init_height
    obs, success = linear_action(env, target_state, render=RENDER)
    return obs, success

def random_and_record(env):
    obs = env._get_observations()
    r = Rotation.from_quat(obs["pot_quat"])
    prev_eul = r.as_euler('xyz')
    target_state = get_current_state(obs)
    action_deltas = np.random.uniform(np.full(3, -0.075), np.full(3, 0.075))
    target_state[0] += action_deltas
    obs, success = linear_action(env, target_state, thresh=0.01, render=RENDER)
    r = Rotation.from_quat(obs["pot_quat"])
    curr_eul = r.as_euler('xyz')
    delta_eul = curr_eul-prev_eul # this is the representation to save
    return [action_deltas, delta_eul]

def full_pot_reset(env):
    env.reset()
    obs, success = init_pot(env)
    initial_right_arm_pos = get_current_state(obs)[2]
    return obs, success, initial_right_arm_pos

if __name__ == "__main__":

    env = get_env()
    obs, success = init_pot(env)
    initial_right_arm_pos = get_current_state(obs)[2]

    data = []
    num_datapoints = 60

    for i in range(num_datapoints):
        if i % 10 == 0:
            print(f"action {i}")
        obs = env._get_observations()

        # also check the right arm position
        target_state = get_current_state(obs)
        curr_right_arm_pos = target_state[2]

        if np.linalg.norm(initial_right_arm_pos - curr_right_arm_pos) > 0.2:
            print("ARM SHIFTED")
            obs, success, initial_right_arm_pos = full_pot_reset(env)
        elif np.linalg.norm(obs["gripper0_to_handle0"]) > 0.1 or np.linalg.norm(obs["gripper1_to_handle1"]) > 0.1:
            print("POT FELL")
            obs, success, initial_right_arm_pos = full_pot_reset(env)

        datapoint = random_and_record(env)
        data.append(datapoint)

    np.save("pot_eul_data", np.array(data))
