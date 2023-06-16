
from robosuite.utils.input_utils import *
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply, axisangle2quat
from robosuite.utils.binding_utils import MjSimState
import numpy as np 
from robosuite.models.base import MujocoModel


def get_env(dilated=False):
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    controller_config["output_max"] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    controller_config["output_min"] = np.array([-0.25, -0.25, -0.25, -0.25, -0.25, -0.25])
    return suite.make(
        env_name="TwoArmPegInHole",
        robots=["UR5e", "UR5e"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["birdview", "agentview"],
        control_freq=20, 
        peg_radius=(0.0075, 0.0075) if dilated else (0.015, 0.015),
        dilated=dilated,
    )

def encode_target_state(pos0, quat0, pos1, quat1):
    return np.concatenate((pos0, quat2axisangle(quat0), pos1, quat2axisangle(quat1)))

def get_current_state(obs):
    robot0_pos = obs["robot0_eef_pos"]
    robot0_quat = obs["robot0_eef_quat"]
    robot1_pos = obs["robot1_eef_pos"]
    robot1_quat = obs["robot1_eef_quat"]
    return encode_target_state(robot0_pos, robot0_quat, robot1_pos, robot1_quat)

def inverse_quaternion(quaternion):
    x, y, z, w = quaternion
    return np.array([-x, -y, -z, w])

def linear_action(env, target_state, update_target_state=None, thresh=0.01, max_steps=150, render=True):
    obs = env._get_observations()
    state = get_current_state(obs)
    error = np.linalg.norm(target_state-state)
    steps = 0
    while error > thresh:
        iquat0 = inverse_quaternion(axisangle2quat(state[3:6]))
        iquat1 = inverse_quaternion(axisangle2quat(state[9:]))
        tquat0 = axisangle2quat(target_state[3:6])
        tquat1 = axisangle2quat(target_state[9:])
        action = target_state.copy()
        action -= state
        action[3:6] = quat2axisangle(quat_multiply(tquat0, iquat0))
        action[9:] = quat2axisangle(quat_multiply(tquat1, iquat1))
        action /= np.linalg.norm(action)
        action *= 0.1
        obs, _, _, _ = env.step(action)
        steps += 1
        if steps >= max_steps:
            return obs, False
        if render:
            env.render()
        state = get_current_state(obs)
        if update_target_state: 
            target_state = update_target_state(obs, target_state)
        error = np.linalg.norm(target_state-state)
    return obs, True

def random_reset(env):
    env.reset()
    # emperically found for right orientations - we only control pos in these demos
    q_initial = np.array([-0.72483611, -0.86707242, 2.63227004, 1.37626896, -0.84602705, -3.1417527, -0.49075564, -1.69662925, 2.49409645, -2.36804778, -1.5709395, 2.6507139])
    sim_state = MjSimState(0, q_initial, np.zeros(q_initial.shape[0]))
    env.sim.set_state(sim_state)
    env.sim.forward()
    # random init pos of eefs
    # regions: 
    # peg - [-0.5:0.5, -0.15:-0.35, 1.25:1.65]
    # sqr - [-0.5:0.5, 0.15:0.35, 1.25:1.65]
    obs = env._get_observations(force_update=True)
    robot0_init_pos = np.random.uniform(np.array([-0.05, -0.3, 1.4]), np.array([0.05, -0.2, 1.5]))
    robot0_quat = obs["robot0_eef_quat"]
    robot0_quat = np.random.normal(robot0_quat, np.full(4, 0.05))
    robot1_init_pos = np.random.uniform(np.array([-0.05, 0.2, 1.4]), np.array([0.05, 0.3, 1.5]))
    robot1_quat = obs["robot1_eef_quat"]
    robot1_quat = np.random.normal(robot1_quat, np.full(4, 0.05))
    target_state = encode_target_state(robot0_init_pos, robot0_quat, robot1_init_pos, robot1_quat)
    obs, success = linear_action(env, target_state, render=False)
    if success:
        return obs
    else:
        return random_reset(env)

def check_contact(sim, geoms_1, geoms_2=None):
    """
    Finds contact between two geom groups.
    Args:
        sim (MjSim): Current simulation object
        geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
            a MujocoModel is specified, the geoms checked will be its contact_geoms
        geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
            If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
            any collision with @geoms_1 to any other geom in the environment
    Returns:
        bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
    """
    # Check if either geoms_1 or geoms_2 is a string, convert to list if so
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    elif isinstance(geoms_1, MujocoModel):
        geoms_1 = geoms_1.contact_geoms
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    elif isinstance(geoms_2, MujocoModel):
        geoms_2 = geoms_2.contact_geoms
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        # check contact geom in geoms
        c1_in_g1 = sim.model.geom_id2name(contact.geom1) in geoms_1
        c2_in_g2 = sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = sim.model.geom_id2name(contact.geom2) in geoms_1
        c1_in_g2 = sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False

def corrected_hole_pos(obs, dilated=False):
    hole_pos = obs["hole_pos"].copy()
    if dilated:
        hole_pos[0] -= 0.055
    else:
        hole_pos[0] -= 0.11
    return hole_pos