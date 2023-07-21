import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../.."))

from stanford_su23.latent_stabilize.utils import get_env, get_current_state, linear_action, gripper_action, inverse_quaternion
from robosuite.utils.transform_utils import quat_multiply, quat2axisangle, quat2mat, mat2euler
import numpy as np
from scipy.spatial.transform import Rotation
import math
import time

RENDER = False

def init_pot(env):
    # workspace for left/right arm:
    # min: [-0.25, -0.25/0, 1]
    # max: [0.25, 0/0.25, 1.5]

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
    target_state[0][2] -= 0.075
    target_state[2] += obs["gripper1_to_handle1"]
    target_state[2][2] -= 0.075
    obs, success = linear_action(env, target_state, render=RENDER)
    obs = gripper_action(env, grasp=True, render=RENDER)

    # pick up
    pot_init_height = 0.25
    target_state = get_current_state(obs)
    target_state[0][2] += pot_init_height
    target_state[2][2] += pot_init_height
    obs, success = linear_action(env, target_state, lock_grippers=True, render=RENDER)
    return obs, success

def random_and_record(env):
    obs = env._get_observations()
    r = Rotation.from_quat(obs["pot_quat"])
    prev_eul = r.as_euler('xyz')
    stab_action = np.random.uniform(np.full(3, -1.), np.full(3, 1.))
    stab_action /= np.linalg.norm(stab_action)
    target_state = get_current_state(obs)
    target_state[0] += stab_action*0.1
    obs, success = linear_action(env, target_state, lock_grippers=True, render=RENDER)
    r = Rotation.from_quat(obs["pot_quat"])
    curr_eul = r.as_euler('xyz')
    return [prev_eul, stab_action, curr_eul] # save s, a -> s
    # return [prev_eul, stab_action, curr_eul-prev_eul] # save s, a -> ds

def full_pot_reset(env):
    env.reset()
    obs, success = init_pot(env)
    initial_right_arm_pos = get_current_state(obs)[2]
    return obs, success, initial_right_arm_pos

if __name__ == "__main__":
    env = get_env()
    obs, success = init_pot(env)
    state = get_current_state(obs)
    initial_left_arm_pos = state[0]
    initial_right_arm_pos = state[2]

    data = []
    num_datapoints = 1000

    for i in range(num_datapoints):
        if i % 10 == 0:
            print(f"action {i}")
        obs = env._get_observations()

        # also check the right arm position
        state = get_current_state(obs)
        curr_left_arm_pos = state[0]
        curr_right_arm_pos = state[2]

        if np.linalg.norm(initial_right_arm_pos - curr_right_arm_pos) > 0.2 or np.linalg.norm(initial_left_arm_pos - curr_left_arm_pos) > 0.2:
            print("ARM SHIFTED")
            obs, success, initial_right_arm_pos = full_pot_reset(env)
        elif np.linalg.norm(obs["gripper0_to_handle0"]) > 0.1 or np.linalg.norm(obs["gripper1_to_handle1"]) > 0.1:
            print("POT FELL")
            obs, success, initial_right_arm_pos = full_pot_reset(env)

        datapoint = random_and_record(env)
        data.append(datapoint)

    np.save("sas_test", np.array(data))
