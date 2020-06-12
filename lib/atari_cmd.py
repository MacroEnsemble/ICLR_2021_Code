import gym
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2, A2C
from manager import AtariPolicyManager
from env_wrapper import ActionRemapWrapper
import argparse
import sys, os

from utils import check_duplicate, str_to_skills, set_GPU, mkdirs
import logger
from datetime import datetime


USE_OLD_data=False


parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=str, required=True, help="e.g SeaquestNoFrameSkip-v4")
parser.add_argument("--rl_model", type=str, default='a2c', help="e.g reinforcement learning model (ppo or a2c)")
parser.add_argument("--skills", type=str, required=True, help="e.g [[1,2,3],[4,5,6]]")
parser.add_argument("--logdir", type=str, default=None, help="worker logdir")
parser.add_argument("--duplicate_checker", type=str, default=None, help="pass any string, then it will check the duplicate skills")
parser.add_argument("--output", type=str, default=None, help="where to write the score")
parser.add_argument("--empty_action", type=int, default=-1, help="the represent number of empty action")
parser.add_argument("--train_total_timesteps", type=int, default=int(5e6), help="train_total_timesteps")
parser.add_argument("--duplicate_penalty", type=int, default=-1, help="return penalty when detect duplicat skills")
parser.add_argument("--include_primitive", type=bool, default=True, help="wheather to include primitive, will use only skills if False")
parser.add_argument("--info", type=str, default=None, help="extra info to write in the file")
parser.add_argument("--set_gpu", type=bool, default=False, help="whether to set gpu device automatically")
parser.add_argument("--save_tensorboard", type=bool, default=False, help="whether to save tensorboard")
parser.add_argument("--save_monitor", type=bool, default=True, help="whether to save monitor")
parser.add_argument("--seed", type=int, default=10000, help="global seed")
parser.add_argument("--preserve_model", type=int, default=0, help="global seed")
parser.add_argument("--evaluate_freq", type=int, default=None, help="evaluate every n steps")

args=parser.parse_args()
def get_info(log_path):
    info=dict()
    with open(log_path, 'r') as f:
        content = f.read().split("\n")
        # print(content)
        for ele in content:
            if ":" in ele:
                key_value = ele.split(":")
                key = key_value[0]
                try: 
                    value = float(key_value[1])
                except ValueError:
                    value = key_value[1]

                info.update({key:value})
    return info

if __name__ == "__main__":
    if args.set_gpu:
    	set_GPU()
    
    
    try:
        

        print('receive task: {}'.format(args))
        
        env_id = args.env_id 
        skills = str_to_skills(args.skills) 
        if args.logdir is not None:
            logdir = os.path.join(args.logdir, str(skills))
        duplicate_checker = args.duplicate_checker
        empty_action = args.empty_action
        train_total_timesteps = args.train_total_timesteps
        duplicate_penalty = args.duplicate_penalty
        include_primitive = args.include_primitive
        evaluate_freq = args.evaluate_freq
        
        logger.Config.Use(filename=os.path.join(args.logdir, "error_log.txt"), level='DEBUG', colored=True)
        LOG = logger.getLogger('main')
       


        print("[train_total_timesteps]:{}".format(train_total_timesteps))
        ave_score = None
        if logdir is not None:
            if os.path.exists(logdir):
                idx = 0
                while True:
                    # use past training score
                    log_file = os.path.join(logdir,"log.txt")

                    # TODO use old data with predefined path
                    if os.path.exists(log_file) and USE_OLD_data:
                        info = get_info(log_file)
                        ave_score = info["ave_score"]
                        break
                    if os.path.exists(logdir + "_{}".format(idx)):
                        log_file = os.path.join(logdir + "_{}".format(idx),"log.txt")
                        if os.path.exists(log_file) and USE_OLD_data:
                            info = get_info(log_file)
                            ave_score = info["ave_score"]
                            break
                        idx = idx + 1
                    else:
                        logdir = logdir + "_{}".format(idx)
                        break
        
        output_path = None
        if args.output is None:
            output_path=os.path.join(logdir, "../", 'output_score')
            mkdirs(output_path)
        else:
            output_path = args.output

        start_time = datetime.now()
        if ave_score is None:
           
            env_creator = lambda env:ActionRemapWrapper(env)
            
            rl_model = None
            if args.rl_model.lower() == "a2c":
                rl_model = A2C
            elif args.rl_model.lower() == "ppo":
                rl_model = PPO2
            else:
                raise NotImplementedError("rl_model invalid: {}".format(args.rl_model))
            atari_manager = AtariPolicyManager(env_id=env_id, env_creator=env_creator, model=rl_model, policy=CnnPolicy, save_path = logdir, verbose=0, num_cpu=20, save_tensorboard=args.save_tensorboard, save_monitor=args.save_monitor,seed=args.seed, preserve_model=args.preserve_model, evaluate_freq=evaluate_freq)
            
            
            skills = list(map(lambda skill:list(filter(lambda x: x!=empty_action, skill)), skills))
            skills = list(filter(lambda skill:len(skill)>1, skills))
            
            
            if duplicate_checker is not None:
                check_skills = skills[:]
                if include_primitive is True:
                    if "health_gathering" in env_id:
                        # env = env_creator(AtariDoom("health_gathering"))
                        raise NotImplementedError
                    else:
                        env = env_creator(gym.make(env_id))
                    primitive_action = [[i] for i in range(env.action_space.n)]
                    check_skills.extend(primitive_action)
                    env.close()
                if check_duplicate(check_skills) is True:
                    ave_score = duplicate_penalty
                else:
                    ave_score, ave_action_reward = atari_manager.get_rewards(skills, train_total_timesteps=train_total_timesteps)            
            else:
                ave_score, ave_action_reward = atari_manager.get_rewards(skills, train_total_timesteps=train_total_timesteps)            
        delta = datetime.now() - start_time

        result = ave_score
        print('result: {}'.format(result))
        filename = "{}_score.txt".format(os.environ['SSHEX_PROC_NUM'])
        with open(os.path.join(output_path,filename),'w') as f:
            print("{}:{}".format(args.skills, result), file=f)
            print("{}:{}".format("skill", args.skills), file=f)
            print("{}:{}".format("score", result), file=f)
            print("{}:{}".format("delta", delta), file=f)
            if args.info is not None:
                print("{}:{}".format("info", args.info), file=f)
            
        # return result
    except:
        LOG.exception('atari_cmd')
        

    
