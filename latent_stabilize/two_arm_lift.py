from stanford_su23.latent_stabilize.utils import get_env, get_current_state, linear_action, gripper_action, inverse_quaternion
from robosuite.utils.transform_utils import quat_multiply, quat2axisangle
import numpy as np
import math

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
    target_state[2] += obs["gripper1_to_handle1"]
    target_state[2][2] -= 0.05
    obs, success = linear_action(env, target_state)
    obs = gripper_action(env, grasp=True)

    target_pot_quat = obs["pot_quat"]

    # generate random traj for right arm
    # enable reacting mode for left arm 
    target_state = get_current_state(obs)
    # target_state[2][2] += np.random.uniform(0.1, 0.15)
    target_state[2][2] += 0.125
    # target_state[2] = np.random.uniform(np.array([-0.25, 0, 1]), np.array([0.25, 0.25, 1.5]))
    obs, success = linear_action(env, target_state, max_steps=1000)

    # move left arm to return pot back to original orientation
    pot_quat = obs["pot_quat"]
    quat_delta = quat_multiply(pot_quat, inverse_quaternion(target_pot_quat))
    angle = np.linalg.norm(quat2axisangle(quat_delta))
    z_offset = pot_width * math.sin(angle)
    target_state = get_current_state(obs)
    target_state[0][2] += z_offset
    obs, success = linear_action(env, target_state, thresh=0.01)

    for i in range(10000):
        env.render()





