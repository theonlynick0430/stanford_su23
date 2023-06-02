import robosuite as suite
from robosuite.utils.input_utils import *
from vis_bcrnn_data_collection_wrapper import BCRNN_DataCollectionWrapper
from robosuite.utils.transform_utils import quat2axisangle, euler2mat, mat2quat, quat_multiply, axisangle2quat, convert_quat
from robosuite.utils.binding_utils import MjSimState
import argparse
import numpy as np 
from robosuite.models.base import MujocoModel
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


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

def random_reset(env):
    env.reset()
    # emperically found for right orientations - we only control pos in these demos
    q_initial = np.array([-0.72483611, -0.86707242, 2.63227004, 1.37626896, -0.84602705, -3.1417527, -0.49075564, -1.69662925, 2.49409645, -2.36804778, -1.5709395, 2.6507139])
    sim_state = MjSimState(0, q_initial, np.zeros(q_initial.shape[0]))
    env.sim.set_state(sim_state)
    env.sim.forward()
    # reseting js does not reset controller so find obs manually 
    # random init pos of eefs
    # regions: 
    # peg - [-0.5:0.5, -0.15:-0.35, 1.25:1.65]
    # sqr - [-0.5:0.5, 0.15:0.35, 1.25:1.65]
    peg_init_pos = np.random.uniform(np.array([-0.25, -0.3, 1.35]), np.array([0.5, -0.2, 1.55]))
    peg_quat = convert_quat(env.sim.data.get_body_xquat(env.robots[0].robot_model.eef_name), to="xyzw")
    peg_quat = np.random.normal(peg_quat, np.full(4, 0.1))
    sqr_init_pos = np.random.uniform(np.array([-0.25, 0.2, 1.35]), np.array([0.25, 0.3, 1.55]))
    sqr_quat = convert_quat(env.sim.data.get_body_xquat(env.robots[1].robot_model.eef_name), to="xyzw")
    sqr_quat = np.random.normal(sqr_quat, np.full(4, 0.1))
    target_state = encode_target_state(peg_init_pos, peg_quat, sqr_init_pos, sqr_quat)
    obs = linear_action(env, target_state, record=False)
    print("loaded random reset")
    env.render()
    return obs

def collect_demo(env, sqr_radius):

    # obs = random_reset(env)
    # while check_contact(env.sim, env.robots[0].robot_model) or check_contact(env.sim, env.robots[1].robot_model):
    #     obs = random_reset(env)

    obs = random_reset(env)

    # x, y, z -> out of screen, x-axis, y-axis
    sqr_target_pos = np.array([0, 0.25, 1.45])
    sqr_target_pos = np.random.normal(sqr_target_pos, 1e-2)
    sqr_target_quat = mat2quat(euler2mat(np.array([-np.pi, 0, 0])))
    peg_target_pos = sqr_target_pos.copy()
    peg_target_pos[1] = -peg_target_pos[1]
    peg_target_pos[2] -= sqr_radius
    peg_rot_quat = mat2quat(euler2mat(np.array([-np.pi/2, -np.pi, 0])))
    peg_target_quat = quat_multiply(peg_rot_quat, sqr_target_quat)
    target_state = encode_target_state(peg_target_pos, peg_target_quat, sqr_target_pos, sqr_target_quat)
    obs = linear_action(env, target_state)
    peg_to_hole = obs["peg_to_hole"]
    target_state = get_current_state(obs)
    adjust = np.array([-0.1, 0, 0])
    target_state[:3] += peg_to_hole+adjust
    obs = linear_action(env, target_state)
    env.flush()

def playback_demo(env, eps_file):
    """Playback data from an episode.

    Args:
        env (MujocoEnv): environment instance to playback trajectory in
        eps_file (str): The path to the file containing data for an episode.
    """
    data = np.load(eps_file)

    env.reset()
    # emperically found for right orientations - we only control pos in these demos
    q_initial = np.array([-0.72483611, -0.86707242, 2.63227004, 1.37626896, -0.84602705, -3.1417527, -0.49075564, -1.69662925, 2.49409645, -2.36804778, -1.5709395, 2.6507139])
    sim_state = MjSimState(0, q_initial, np.zeros(q_initial.shape[0]))
    env.sim.set_state(sim_state)
    env.sim.forward()
    env.render()

    # ee_position = data["ee_position"]
    # ee_orientation = data["ee_orientation"]
    # for pos, ori in zip(ee_position, ee_orientation):
    #     target_state = encode_target_state(pos[:3], ori[:4], pos[3:], ori[4:])
    #     linear_action(env, target_state, record=False)

    actions = data["action"]
    for action in actions:
        env.step(action)
        env.render()


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

def linear_action(env, target_state, thresh=0.05, record=True):
    obs, _, _, _ = env.step(np.zeros(env.action_dim), record=False)
    state = get_current_state(obs)
    error = np.linalg.norm(target_state-state)
    while error > thresh:
        iquat0 = inverse_quaternion(axisangle2quat(state[3:6]))
        iquat1 = inverse_quaternion(axisangle2quat(state[9:]))
        tquat0 = axisangle2quat(target_state[3:6])
        tquat1 = axisangle2quat(target_state[9:])
        action = target_state.copy()
        action -= state
        action[3:6] = quat2axisangle(quat_multiply(tquat0, iquat0))
        action[9:] = quat2axisangle(quat_multiply(tquat1, iquat1))
        if np.linalg.norm(action) > 0.1:
            action /= np.linalg.norm(action)
            action *= 0.1
        obs, _, _, _ = env.step(action, record=record)
        if record:
            env.render()
        state = get_current_state(obs)
        error = np.linalg.norm(target_state-state)
    return obs

    # noise = 0
    # step = 5e-2
    # num_waypoints = int(np.linalg.norm(target_state-state)/step)
    # vec = target_state-state
    # for i in range(1, num_waypoints+1):
    #     waypoint = state + (target_state-state)*i/num_waypoints
    #     if i % 2:
    #         waypoint += np.random.normal(0, noise, size=waypoint.shape)
    #     obs, _, _, _ = env.step(waypoint)
    #     state = get_current_state(obs)
    #     env.render()
    # obs, _, _, _ = env.step(target_state)
    # env.render()
    # return obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="tmp/")
    parser.add_argument("--demos", type=int, default=1)
    parser.add_argument("--playback", type=str, default=None)
    parser.add_argument("--dilated", action="store_true")
    args = parser.parse_args()

    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    controller_config["output_max"] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    controller_config["output_min"] = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    
    peg_radius = (0.015, 0.015)
    sqr_radius = 0.17
    if args.dilated:
        peg_radius = (0.0075, 0.0075)
        sqr_radius = 0.085
    env = suite.make(
        env_name="TwoArmPegInHole",
        robots=["UR5e", "UR5e"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["birdview", "agentview"],
        control_freq=20, 
        peg_radius=peg_radius,
        dilated=args.dilated,
    )

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
    