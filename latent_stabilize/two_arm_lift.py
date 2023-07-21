import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../.."))

from stanford_su23.latent_stabilize.utils import get_env, get_current_state, linear_action, gripper_action, inverse_quaternion
from robosuite.utils.transform_utils import quat_multiply, quat2axisangle, quat2mat, mat2euler
import numpy as np
import math
import time

def eul_delta(quat, target_quat):
    # returns target - current
    quat_delta = quat_multiply(target_quat, inverse_quaternion(quat))
    eul_delta = mat2euler(quat2mat(quat_delta)) # safe because delta is so small that we don't get gymbal lock
    return eul_delta

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
    # target_state[0][2] -= 0.05
    target_state[0][2] -= 0.075
    target_state[2] += obs["gripper1_to_handle1"]
    # target_state[2][2] -= 0.05
    target_state[2][2] -= 0.075
    obs, success = linear_action(env, target_state)
    obs = gripper_action(env, grasp=True)

    # pick up
    pot_init_height = 0.25
    target_state = get_current_state(obs)
    target_state[0][2] += pot_init_height
    target_state[2][2] += pot_init_height
    obs, success = linear_action(env, target_state)

    target_pot_quat = obs["pot_quat"]

    # for i in range(3):
    prev_action = None
    num_iter = 2
    # action_norm = 0.035
    action_norm = 0.015
    actions = np.array([
    [0, action_norm, -action_norm],
    [action_norm, 0, -action_norm],
    [-action_norm, -action_norm, action_norm],
    [-action_norm, -action_norm, 0],
    [action_norm, action_norm, action_norm],
    ])
    for i in range(60):
        print(f"action {i}")
        # generate random traj for right arm
        target_state = get_current_state(obs)
        # target_state[2][0] += np.random.uniform(-0.125, 0.125)
        # target_state[2] += np.random.uniform(np.full(3, -0.125), np.full(3, 0.125))

        # if prev_action is None:
        #     prev_action = np.random.uniform(np.full(3, -0.125), np.full(3, 0.125))
        # target_state[2] += prev_action/num_iter
        # if i % num_iter == num_iter-1: prev_action = None

        # target_state[2] += np.random.uniform(np.full(3, -0.075), np.full(3, 0.075))
        # target_state[2] += np.random.uniform(np.full(3, -0.025), np.full(3, 0.025))

        # acting_action = np.random.uniform(np.full(3, 0.015), np.full(3, 0.05))
        # print("acting first", acting_action)
        # acting_action = np.array([a if a > 0.02 else 0 for a in acting_action])
        # sample = (np.random.binomial(1, 0.5, size=3)-0.5)*2
        # sample[2] = 1
        # print("sample", sample)
        # acting_action = acting_action * sample
        # print("acting action", acting_action)
        # target_state[2] += acting_action

        # target_state[2] += np.array([0.05, 0.05, 0.05])
        target_state[2] += actions[np.random.randint(0, 5)]

        # target_state[2] = np.random.uniform(np.array([-0.25, 0, 1]), np.array([0.25, 0.25, 1.5]))
        obs, success = linear_action(env, target_state, max_steps=1000)

        # move left arm to return pot back to original orientation
        # ASSUMPTIONS:
        # - we need to know the pot width
        # - we need to know which movements correspond to which degrees of freedom:
        #   - pos x-axis: rot z-axis
        #   - pos y-axis: none
        #   - pos (-) z-axis: rot x-axis
        def update_target_state(obs, _):
            ed = eul_delta(obs["pot_quat"], target_pot_quat)
            x_offset = pot_width * math.sin(ed[2])
            z_offset = -pot_width * math.sin(ed[0])
            target_state = get_current_state(obs)
            target_state[0][0] += x_offset
            target_state[0][2] += z_offset
            return target_state
        updated_target = update_target_state(obs, target_pot_quat)
        obs, success = linear_action(env, updated_target, update_target_state=update_target_state, thresh=0.01)
        error = eul_delta(obs["pot_quat"], target_pot_quat)
        print(error)
        # time.sleep(3)

    for i in range(1):
        env.render()
