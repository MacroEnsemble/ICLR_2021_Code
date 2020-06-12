

import argparse
from env_wrapper import SkillWrapper
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv, VecFrameStack
from stable_baselines import PPO2, A2C
from manager import AtariPolicyManager
from env_wrapper import ActionRemapWrapper
from stable_baselines.common import set_global_seeds
import os
import glob
import re
import time
from cmd_util import make_atari_env
import errno
import shutil
from utils import mkdirs


import scipy.misc
from PIL import Image
import sys
sys.path.append(os.path.abspath("../DQN_skill"))
from constant import env_list as ENV_LIST

parser = argparse.ArgumentParser()

parser.add_argument("load_model", type=str, default=None, help="If the model path contains enough info, this is the only necessary argument.(directory to *.pkl)")
parser.add_argument("--rl_model", type=str, default=None, help="ppo or a2c")
parser.add_argument("--logdir", type=str, default=None, help="save path")
parser.add_argument("--record", action="store_true")
parser.add_argument("--log_action", action="store_true")
parser.add_argument("--log_picture", action="store_true")
parser.add_argument("--seed", type=int, default=2000)
parser.add_argument("--train_total_timesteps", type=int, default=10000000)
parser.add_argument("--eval_max_steps", type=int, default=int(10e6))
args = parser.parse_args()


train_total_timesteps=args.train_total_timesteps

MAX_VIDEO_LENGTH = 1000000
def str_to_skills(str_skills):
    str_skills = str_skills.replace(" ", '')
    
    skills = []
    temp_idx = 0
    str_skills = str_skills[1:-1]
    for idx, ch in enumerate(str_skills):
        if ch=="[":
            temp_idx = idx
        elif ch=="]":
            
            act_seq = str_skills[temp_idx+1:idx].split(",")
            skill = []
            for act in act_seq:
                skill.append(int(act))
            skills.append(skill)
    return skills

def record_():
    model_path = args.load_model
    os.path.isfile(model_path)
    
    # search skills
    
    m=re.search("\[[0-9\, \[\]]*\]", model_path)
    if m is  None:
        raise ValueError("load_model: {} does not contain skills".format(model_path))
    skills = str_to_skills(m.group(0))
    

    
    # search env-id
    env_id_list = ENV_LIST
    env_id=None
    searched = False
    m = re.search("[A-Z][a-z]*NoFrameskip-v4", model_path)
    if m is not None:
        searched = True
        env_id = m.group(0)
    
    if searched is not True:
        for id_ in env_id_list:
            if  id_.lower() in model_path.lower():
                searched = True
                env_id = id_ + "NoFrameskip-v4"

    if searched is not True:
        raise ValueError("load_model: {} does not contain env id".format(model_path))
    
    save_path = args.logdir
    if save_path is None:
        save_path = os.path.dirname(model_path)
    
    print("ENV:{} \nskills:{} \nmodel_path:{} \nsave_path:{}\n".format(env_id, skills, model_path, save_path))
    time.sleep(3)


    env_creator_ = lambda env:ActionRemapWrapper(env)
    env_creator = lambda env:SkillWrapper(env_creator_(env), skills=skills)
    env = VecFrameStack(make_atari_env(env_id, 1, args.seed, extra_wrapper_func=env_creator, logdir=save_path, wrapper_kwargs={"episode_life":False, "clip_rewards":False}), 4)

    
    if args.load_model is None:
        raise NotImplementedError
    assert os.path.isfile(args.load_model)

    if args.rl_model == "ppo":
        model = PPO2.load(args.load_model)
    elif args.rl_model == "a2c":
        model = A2C.load(args.load_model)
    elif args.rl_model is None:
        if "ppo" in model_path:
            model = PPO2.load(model_path)  
        elif "a2c" in model_path:
            model = A2C.load(model_path)
        else:
            raise ValueError("please specify rl_model")
    else:
        raise ValueError("{} rl_model not recognize".format(args.rl_model))

    # DEBUG
    set_global_seeds(args.seed)
    
    obs = env.reset()
    if args.record:
        env = VecVideoRecorder(env, save_path, record_video_trigger=lambda x: x == 0, video_length=MAX_VIDEO_LENGTH)
        env.reset()
    total_rewards = 0 

    
    action_save_path=os.path.join(save_path, "history_action.txt")
    if args.log_action:
        try:
            os.remove(action_save_path)
        except OSError as e:
            if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                raise # re-raise exception if a different error occurred
    log_picture = None
    if args.log_picture:
        log_picture = os.path.join(save_path, "history_action_pic")
        log_picture = mkdirs(log_picture, mode="keep")
        action_save_path = os.path.join(log_picture, os.path.basename(action_save_path))
        # try:
        #     # shutil.rmtree()
        # except:

    print("start evaluate")
    with open(action_save_path, 'a') as f:
        for steps in range(args.eval_max_steps):
            action, _states = model.predict(obs)
            if args.log_action:
                # print("{}".format(action[0]), sep=" ", file=f)
                f.write("{} ".format(action[0]))
            if args.log_picture:
                assert log_picture is not None
                pict = env.render(mode='rgb_array')
                
                im = Image.fromarray(pict)
                _path = os.path.join(log_picture, "{}_{}.jpg".format(steps, action[0]))
                im.save(_path)
            obs, rewards, dones, info = env.step(action)
            total_rewards += rewards
            if bool(dones[0]) is True:
                break  
    print("steps: {}/{}".format(steps+1, args.eval_max_steps))
    print("total_rewards: {}".format(total_rewards))
    env.close()





if __name__ == "__main__":
    # test_search()
    record_()
