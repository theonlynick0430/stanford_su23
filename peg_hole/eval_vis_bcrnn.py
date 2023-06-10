"""
Evaluate a policy and model in an environment. No saving of data (see scripts/collect.py)
"""

import os
import torch
from configs.helpers import get_script_parser, load_base_config
from muse.experiments.file_manager import ExperimentFileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.general_utils import exit_on_ctrl_c
from attrdict import AttrDict as d
from stanford_su23.peg_hole.utils import get_env
from stanford_su23.eval_vis_bcrnn_wrapper import BCRNN_ModelEvaluationWrapper
from muse.utils.torch_utils import to_numpy
from stanford_su23.peg_hole.utils import get_env, linear_action, random_reset, encode_target_state, get_current_state, check_contact


if __name__ == "__main__":
    exit_on_ctrl_c()

    # things we can use from command line
    parser = get_script_parser()
    parser.add_argument('config', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--max_eps', type=int, default=0)
    parser.add_argument("--dilated", action="store_true")
    args, unknown = parser.parse_known_args()

    # load the config
    params, root = load_base_config(args.config, unknown)
    exp_name = root.get_exp_name()

    # file_manager = ExperimentFileManager(exp_name, is_continue=True)

    model_fname = args.model_file
    # model_fname = file_path_with_default_dir(model_fname, file_manager.models_dir)
    # assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    # print("Using model: {}".format(model_fname))

    # generate env
    env_spec = params.env_spec.cls(params.env_spec)
    env = get_env(dilated=args.dilated)
    env = BCRNN_ModelEvaluationWrapper(env)

    # generate model and policy
    model = params.model.cls(params.model, env_spec, None)
    policy = params.policy.cls(params.policy, env_spec, env=env)

    # reset the environment and policy
    random_reset(env)
    # while check_contact(env.sim, env.robots[0].robot_model) or check_contact(env.sim, env.robots[1].robot_model):
    print("loaded random reset")
    env.render()
    policy.reset_policy()

    # restore model from file
    model.restore_from_file(model_fname)

    # pytorch eval mode
    model.eval()

    # actual eval loop
    ep = 0
    i = 0 
    steps = 0
    while True:
        # TODO: check for success or eps terminated by robosuite
        if policy.is_terminated(model, env.get_input(), d()):
            print(f"[{ep}] Resetting env after {i} steps")
            steps += i
            i = 0
            ep += 1
            # terminate condition
            if ep >= args.max_eps > 0:
                print("max eps reached...exiting")
                break
            random_reset(env)
            policy.reset_policy()

        # empty axes for (batch_size, horizon)
        obs = env.get_input()
        expanded_obs = obs.leaf_apply(lambda arr: arr[:, None])
        with torch.no_grad():
            action = policy.get_action(model, expanded_obs, d())

        # step the environment with the policy action
        base_action = to_numpy((action.action)[0][0], check=True)
        print("base_action")
        print(base_action)
        obs = env.step(base_action)
        env.render()
        i += 1