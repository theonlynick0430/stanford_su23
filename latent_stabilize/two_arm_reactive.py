import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../.."))

from stanford_su23.latent_stabilize.utils import get_env, get_current_state, linear_action, single_action, gripper_action, inverse_quaternion
from robosuite.utils.transform_utils import quat_multiply, quat2axisangle, quat2mat, mat2euler
import numpy as np
import math
import time

def eul_delta(quat, target_quat):
    # returns target - current
    quat_delta = quat_multiply(target_quat, inverse_quaternion(quat))
    eul_delta = mat2euler(quat2mat(quat_delta)) # safe because delta is so small that we don't get gymbal lock
    return eul_delta

def linear_reactive(env, target_state, update_target_state=None, thresh=0.05, max_steps=100, render=True):
    obs = env._get_observations()
    state = get_current_state(obs)
    error = np.linalg.norm(np.concatenate(target_state)-np.concatenate(state))
    steps = 0
    stable_end = 0
    while error > thresh and stable_end < 5:
        # for orientation, env takes in delta axis-angle commands relative to world axis
        tquat0 = quat_multiply(target_state[1], inverse_quaternion(state[1]))
        tquat1 = quat_multiply(target_state[3], inverse_quaternion(state[3]))
        action = np.concatenate((target_state[0]-state[0], quat2axisangle(tquat0), [0],
                                 target_state[2]-state[2], quat2axisangle(tquat1), [0]))
        action /= np.linalg.norm(action)
        action *= 0.02
        obs, _, _, _ = env.step(action)
        steps += 1
        print("steps", steps)
        if steps >= max_steps:
            stable_end += 1
        if render:
            env.render()
        state = get_current_state(obs)
        if update_target_state:
            target_state = update_target_state(obs, target_state)
        error = np.linalg.norm(np.concatenate(target_state)-np.concatenate(state))
    return obs, True

if __name__ == "__main__":

    env = get_env()

    # workspace for left/right arm:
    # min: [-0.25, -0.25/0, 1]
    # max: [0.25, 0/0.25, 1.5]

    # obs = gripper_action(env, grasp=True)
    # target_state = get_current_state(obs)
    # print(target_state[0][2])
    # target_state[0][2] = 1.25
    # linear_action(env, target_state, max_steps=1000)

    # find width of pot
    obs = env._get_observations(force_update=True)
    pot_width = np.linalg.norm(obs["handle0_xpos"]-obs["handle1_xpos"])

    # grasp handles
    obs = gripper_action(env, grasp=True)
    rot_n90 = np.array([0, 0, -math.sqrt(2)*0.5, math.sqrt(2)*0.5])
    target_state = get_current_state(obs)
    target_state[0] += obs["gripper0_to_handle0"]
    target_state[0][2] += 0.1
    target_state[1] = quat_multiply(rot_n90, quat_multiply(obs["pot_quat"], target_state[1]))
    target_state[2] += obs["gripper1_to_handle1"]
    target_state[2][2] += 0.1
    target_state[3] = quat_multiply(rot_n90, quat_multiply(obs["pot_quat"], target_state[3]))
    obs, success = linear_action(env, target_state, max_steps=1000)
    obs = gripper_action(env, grasp=False)
    target_state = get_current_state(obs)
    target_state[0] += obs["gripper0_to_handle0"]
    target_state[0][2] -= 0.05
    # target_state[0][2] -= 0.075
    target_state[2] += obs["gripper1_to_handle1"]
    target_state[2][2] -= 0.05
    # target_state[2][2] -= 0.075
    obs, success = linear_action(env, target_state)
    obs = gripper_action(env, grasp=True)

    target_pot_quat = obs["pot_quat"]

    # pick up
    pot_init_height = 0.25
    target_state = get_current_state(obs)
    target_state[0][2] += pot_init_height
    target_state[2][2] += pot_init_height
    obs, success = linear_action(env, target_state)

    # target_pot_quat = obs["pot_quat"]

    def update_target_state(obs, target_state):
        ed = eul_delta(obs["pot_quat"], target_pot_quat)
        # first we need to know where the grasp is relative to the object center

        # then we can convert that into a coordinate

        # then we can transform the coordinate into the new frame

        
        x_offset = pot_width * math.sin(ed[2])
        z_offset = -pot_width * math.sin(ed[0])
        target_state[0][0] += x_offset
        target_state[0][2] += z_offset
        return target_state


    # naive updated
    # def update_target_state(obs, target_state):
    #     ed = eul_delta(obs["pot_quat"], target_pot_quat)
    #     x_offset = pot_width * math.sin(ed[2])
    #     z_offset = -pot_width * math.sin(ed[0])
    #     target_state[0][0] += x_offset
    #     target_state[0][2] += z_offset
    #     return target_state

    for i in range(60):
        print(f"action {i}")
        # generate random traj for right arm
        target_state = get_current_state(obs)
        deltas = np.random.uniform(np.full(3, -0.075), np.full(3, 0.075))
        # deltas = np.array([0, 0, 0.1])
        print("deltas", deltas)
        target_state[2] += deltas
        obs, success = linear_reactive(env, target_state, max_steps=1000, update_target_state=update_target_state)


    for i in range(1):
        env.render()
