
from robosuite.utils.input_utils import *
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply
import numpy as np
import robosuite as suite
from robosuite.utils.binding_utils import MjSimState

GRIPPER_CLOSED = 1
GRIPPER_OPEN = -1

FINGER0_CLOSED_JS_UR = 4.99108192e-01
FINGER1_CLOSED_JS_UR = -2.16154334e-01

def get_env():
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    controller_config["output_max"] = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5])
    controller_config["output_min"] = np.array([-0.25, -0.25, -0.25, -0.5, -0.5, -0.5])
    return suite.make(
        env_name="TwoArmLift",
        robots=["UR5e", "UR5e"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["birdview", "agentview"],
        control_freq=20,
        horizon=10500,
    )

def get_current_state(obs):
    robot0_pos = obs["robot0_eef_pos"].copy()
    robot0_quat = obs["robot0_eef_quat"].copy()
    robot1_pos = obs["robot1_eef_pos"].copy()
    robot1_quat = obs["robot1_eef_quat"].copy()
    return [robot0_pos, robot0_quat, robot1_pos, robot1_quat]

def inverse_quaternion(quaternion):
    x, y, z, w = quaternion
    return np.array([-x, -y, -z, w])

def linear_action(env, target_state, update_target_state=None, lock_grippers=False, thresh=0.05, max_steps=20, render=True):
    obs = env._get_observations()
    state = get_current_state(obs)
    error = np.linalg.norm(np.concatenate(target_state)-np.concatenate(state))
    steps = 0
    if lock_grippers:
        grip0_idx = np.max(np.array(env.robots[0]._ref_joint_pos_indexes))+1
        grip1_idx = np.max(np.array(env.robots[1]._ref_joint_pos_indexes))+1
    while error > thresh:
        # for orientation, env takes in delta axis-angle commands relative to world axis
        tquat0 = quat_multiply(target_state[1], inverse_quaternion(state[1]))
        tquat1 = quat_multiply(target_state[3], inverse_quaternion(state[3]))
        action = np.concatenate((target_state[0]-state[0], quat2axisangle(tquat0), [0],
                                 target_state[2]-state[2], quat2axisangle(tquat1), [0]))
        obs, _, _, _ = env.step(action) 
        if lock_grippers:
            qpos = env.sim.data.qpos
            qpos[grip0_idx] = FINGER0_CLOSED_JS_UR
            qpos[grip0_idx+1] = FINGER1_CLOSED_JS_UR
            qpos[grip1_idx] = FINGER0_CLOSED_JS_UR
            qpos[grip1_idx+1] = FINGER1_CLOSED_JS_UR
            qvel = env.sim.data.qvel
            qvel[grip0_idx] = 0
            qvel[grip0_idx+1] = 0
            qvel[grip1_idx] = 0
            qvel[grip1_idx+1] = 0
            sim_state = MjSimState(0, qpos, qvel)
            env.sim.set_state(sim_state)
            env.sim.forward()
        steps += 1
        if steps >= max_steps:
            return obs, False
        if render:
            env.render()
        state = get_current_state(obs)
        if update_target_state:
            target_state = update_target_state(obs, target_state)
        error = np.linalg.norm(np.concatenate(target_state)-np.concatenate(state))
    return obs, True

def gripper_action(env, grasp=True, render=True):
    action = np.zeros(14)
    if grasp:
        action[6] = GRIPPER_CLOSED
        action[13] = GRIPPER_CLOSED
    else:
        action[6] = GRIPPER_OPEN
        action[13] = GRIPPER_OPEN
    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(action)
        if render:
            env.render()
    return obs
