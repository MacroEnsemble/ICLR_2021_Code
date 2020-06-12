# demo code for AtariPolicyManager in manager.py
from env_wrapper import SkillWrapper
import gym
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from manager import AtariPolicyManager ##DEBUG TEST
from env_wrapper import ActionRemapWrapper
import argparse
import sys, os

sys.path.append(os.path.abspath("../NAO_skill_search/nao_search/nao_search/"))
sys.path.append(os.path.abspath("../DQN_skill"))
from constant import env_list
from utils import check_duplicate

import multiprocessing
import time
import GPUtil
import shutil
import itertools
import errno


sys.path.append(os.path.abspath("../NAO-skill-search"))
import sshex, sshex_c, sshex.logger

sys.path.append(os.path.abspath("../DQN_skill"))
from generate_config_adqn import gen_config

def wait_GPU(delay_time=30):
    while True:
        deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
        if len(deviceIDs)==0:
            print("No GPU available, wait for {} sec".format(delay_time))
            time.sleep(delay_time)
        else:
            print("export CUDA_VISIBLE_DEVICES={}".format(deviceIDs[0]))
            os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(deviceIDs[0])
            break
#TODO
# change to multiprocesss
# from nao_search.processes import UnsafePool

# ENV = "Alien-ramDeterministic-v4"
# ENV="SeaquestNoFrameskip-v4"
LOGDIR = "../log/other/all_env_log_ori/"
# ENV_LIST = ["Alien","Seaquest","BeamRider","SpaceInvaders", "Qbert", "Pong", "Enduro", "Breakout", "Defender", "Phoenix", "KungFuMaster", "MsPacman", "Venture", "Freeway", "Amidar", "Atlantis", "Asteroids", "Gravitar", "Frostbite", "Solaris", "CrazyClimber"]
ENV_LIST = env_list


# rl_model = "ppo"
# assert args.env_id in ENV_LIST
## Qbert




USE_OLD_data=False
GAMMA=0.9
SEED=60001
# TEST_TIMES = 5
TRAIN_TOTAL_TIMESTEPS=None


# TRAIN_TOTAL_TIMESTEPS = int(10e7)
# TRAIN_TOTAL_TIMESTEPS = int(100)
# EVALUATE_FREQ=250000
EVALUATE_FREQ=None




def mkdirs(path, keep=True):
    # for i in itertools.count():
    idx = 0
    while True:
        try:
            os.makedirs(path)
            break
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                if keep is True:
                    path = "{}_{}".format(path, idx)
                    idx += 1
                else:
                    while True:
                        ans=input("Are you sure to remove {} (y/n)? ".format(path))
                        if ans.lower()=='y':
                            shutil.rmtree(path, ignore_errors=True)
                            break
                        elif ans.lower()=='n':
                            raise OSError("{} exists".format(path))
            elif exc.errno==errno.EBUSY:
                print("resource device busy, wait 5 sec and retry...")
                time.sleep(5)
            else:
                raise
    return True

def str_to_skills(str_skills):
    str_skills = str_skills.replace(" ", '')
    str_skills = str_skills.replace("[", '')
    str_skills = str_skills.replace("]", '')
    str_skills = str_skills.split(",")
    skills = []
    for act in str_skills:
        skills.append(int(act))
    return skills
# def main(skills):
#     env_creator = lambda:ActionRemapWrapper(gym.make(args.env_id))
#     atari_manager = AtariPolicyManager(env_creator=env_creator, model=PPO2, policy=MlpPolicy, save_path = args.logdir, verbose=1, num_cpu=15)
#     ave_score, ave_action_reward = atari_manager.get_rewards(str_to_skills(args.skills))
#     return ave_score


# === create sshex ===


def evaluate(env_dict, execute_dir, eval_times, train_total_timesteps, evaluate_freq, server_names,dry_run=False):
    env_name = env_dict["env_id"]
    if "NoFrameskip-v4" in env_name:
        env_name = env_name.replace("NoFrameskip-v4", "")
    assert env_name in ENV_LIST
    env_id = env_name + "NoFrameskip-v4"
    # if "NoFrameskip-v4" not in env_id:
    #     env_id = env_id + "NoFrameskip-v4"
    macro_list = env_dict["macro"]
    subdir = env_dict.get("subdir", "./")
    rl_model = env_dict.get("rl_model", "ppo")
    preserve_model = env_dict.get("preserve_model", 1)
    save_tensorboard = env_dict.get("save_tensorboard", 'False')
    train_total_timesteps = env_dict.get('train_total_timesteps', train_total_timesteps)
    # logdir_ = os.path.join(args.logdir, env_id)
    logdir_ = os.path.join(execute_dir, subdir)

    # global_seed = SEED

    # === create config ===
    sshex.logger.Config.Use(filename='{}'.format(os.path.join(execute_dir, "error_log.txt")),
            colored=True,
            level='DEBUG')
    config = gen_config(server_names=server_names, env=env_name)
    sshex_c.Config(config)

    for macros in macro_list:
        command=[]
        macros = str(macros).replace(" ", "")
        global_seed = env_dict.get("seed", SEED)
        eval_times = env_dict.get("eval_times", eval_times)
        for i in range(eval_times):
            # atari_cmd_copy DEBUG TEST
            data = ['source ~/skill_search.sh; ',  'python', 'atari_cmd.py',\
                '--env_id', env_id, '--skill', '{}'.format(macros),\
                '--logdir', logdir_,\
                '--duplicate_checker', 'True', '--empty_action', '-1',\
                '--train_total_timesteps', train_total_timesteps,\
                '--seed', '{}'.format(global_seed),\
                '--preserve_model', '{}'.format(preserve_model),\
                '--save_tensorboard', str(save_tensorboard),\
                '--rl_model', "{}".format(rl_model),\
                '--evaluate_freq', "{}".format(evaluate_freq)]

            command.append(data)

            global_seed += 100
        if dry_run:
            print("==========")
            for cmd in command:
                print(cmd)
        else:
            sshex_c.map(command, wait=False, retry=True, cooldown=30)


from atari_cmd_list import get_cmd_from_exp_id
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=str, nargs='+')
    parser.add_argument("--logdir", type=str, default="./")
    parser.add_argument("--server", '-s', type=str, default="s", help="server name")
    parser.add_argument("--eval_times", type=int, default=6, help="evaluate how many times")
    parser.add_argument("--train_total_timesteps", type=int, default=int(1.001e7))
    parser.add_argument("--eval_freq", type=int, default=50000, help="evaluate every n steps")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    TRAIN_TOTAL_TIMESTEPS=args.train_total_timesteps
    EVALUATE_FREQ=args.eval_freq
    server_names = args.server
    

    for exp_id in args.exp_id:
        test_env = get_cmd_from_exp_id(exp_id)
        for env_dict in test_env:
            # wait_GPU()
            evaluate(env_dict, args.logdir, args.eval_times, args.train_total_timesteps, EVALUATE_FREQ, server_names, args.dry_run)
