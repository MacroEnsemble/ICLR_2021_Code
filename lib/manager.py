import gym
import time
from env_wrapper import SkillWrapper
from collections import deque, OrderedDict, Counter
import os
import errno
import datetime
import yaml
import shutil
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common import set_global_seeds
import numpy as np
import glob
import time
import shutil
import yaml 
from cmd_util import make_atari_env, make_doom_env
from utils import mkdirs



class AtariPolicyManager(object):
    def __init__(self, env_id, env_creator, model, policy, save_path, preserve_model=0, num_cpu=15, pretrain=False, log_action_skill=True, save_tensorboard=True, save_monitor=True, restore_training=False, verbose=0, seed=1000, gamma=0.99, evaluate_freq=None, use_converge_parameter=False):
        """
        :(deprecate)param env: (gym.core.Env) gym env with discrete action space
        :env_creator: (lambda function) environment constructor to create an env with type (gym.core.Env )
        :param model: any model in stable_baselines e.g PPO2, TRPO...
        :param policy: (ActorCriticPolicy) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
        :param save_path: (str) path to store model and log
        :param preserve_model: (int) how much history model will be preserved
        :param num cpu(num_env): (int) train on how many env
        :param pretrain: (bool) if the model passed in are pretrained
        :param log_action_skill(bool) wheather to count frequency of actions
        :param restore_training(bool) wheather to restore last times training
        :param verbose: (int) 0,1 wheather or not to print the training process
        :param evaluate_freq=None: (int) evaluate model every n steps. If None, evaluate only after training phase is finished.
        """
        # super(AtariPolicyManager, self).__init__()
        # self.env=env
        
        if restore_training:
            assert os.path.exists(os.path.join(save_path, "log.txt"))
            self.save_path = save_path
            self.load_hyperparameter()
        else:
            self.env_id=env_id
            self.preserve_model = preserve_model
            self.verbose = verbose
            self._save_model_name=deque()
            self._serial_num = 1
            self.num_cpu = num_cpu
            self.reset_num_timesteps = not pretrain
            self.seed = seed
            self.gamma=gamma
            self.evaluate_freq=evaluate_freq
            self.use_converge_parameter = use_converge_parameter
            if save_path is None:
                self.save_path = None
            elif os.path.exists(save_path):
                # use exists dir
                self.save_path = os.path.abspath(save_path)

            else:
                self.save_path = save_path
                try:
                    os.makedirs(save_path)
                except OSError as ex:
                    if ex.errno == errno.EEXIST and os.path.exists(save_path):
                        print("[CRTICAL SECTION] create two same dir: {} create".format(save_path))
                        pass
                    else:
                        raise 
            self.save_tensorboard = save_tensorboard
            self.save_monitor = save_monitor
            self.log_action_skill = log_action_skill
            self.save_hyperparameter()
        self.env_creator = env_creator
        self.model = model
        self.policy = policy
    def save_hyperparameter(self):
        if self.save_path is not None:
            filename = os.path.join(self.save_path, "manager_hyperparameter.yml")
            d=dict()
            d.update(self.__dict__)
            print(d)
            with open(filename, 'w') as outfile:
                yaml.dump(d, outfile, default_flow_style=True)
    def load_hyperparameter(self):
        filename = os.path.join(self.save_path, "manager_hyperparameter.yml")
        assert os.path.exists(filename)
        with open(filename, 'r') as readfile:
            d = yaml.load(readfile)
            print(d)
            self.preserve_model = d["preserve_model"]
            self.verbose = d["verbose"]
            self._save_model_name = d["_save_model_name"] 
            # self._serial_num = 1
            self.num_cpu = d["num_cpu"]
            self.reset_num_timesteps = d["_save_model_name"]
            self.save_tensorboard = d["save_tensorboard"]
            self.log_action_skill = d["log_action_skill"]
            self.env_id = d["env_id"]
            self.seed = d["seed"]
            self.gamma=d["gamma"]
            self.evaluate_freq=d["evaluate_freq"]
        with open(os.path.join(self.save_path, "log.txt"), 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if "Episode: " in line:
                    self._serial_num = int(line.replace("Episode: ", ''))

        print(self.__dict__)
        time.sleep(5)
    def make_env(self, env_creator, rank, skills=[], action_table=None,seed=0):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = env_creator()
            env = SkillWrapper(env, skills=skills)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    def evaluate(self, env, model, eval_times, eval_max_steps, render=False):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: info
        """

        # evaluate with multiprocess env
        if self.num_cpu>1:
            episode_rewards = [[] for _ in range(env.num_envs)]
            ep_rew = [0.0 for _ in range(env.num_envs)]

            action_statistic = OrderedDict()
            for i in range(env.action_space.n):
                    action_statistic[str(env.action_space[i])]=0
            act_log = [[]for _ in range(env.num_envs)]

            obs = env.reset()
            # for i in range(num_steps):
            ep_count = 0
            total_actions_count=0
            print("start to eval agent...")
            
            for j in range(eval_max_steps):
                # _states are only useful when using LSTM policies
                actions, _states = model.predict(obs)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, rewards, dones, info = env.step(actions)

                # Stats
                for i in range(env.num_envs):
                    
                    ep_rew [i] = ep_rew[i] + rewards[i]
                    act_log[i].append(actions[i])
                    if render:
                        env.render()
                        time.sleep(0.05)
                    if dones[i]:
                        
                        episode_rewards[i].append(ep_rew[i])
                        
                        act_count = Counter(np.asarray(act_log[i]).flatten())
                        for key in act_count:
                            action_statistic[str(env.action_space[key])] +=  act_count[key]
                            total_actions_count += act_count[key]
                        ep_rew[i] = 0
                        act_log[i] = []
                        ep_count = ep_count + 1
                if ep_count >= eval_times:
                    break
            print("Finish eval agent")
            print("Elapsed: {} sec".format(round(time.time()-self.strat_time, 3)))

            
            total_reward = []
            
            # does not meet eval tims:
            if ep_count<eval_times:
                total_reward.extend(ep_rew)
                print("WATCH OUT: does not reach evaluate required times")
                print("current: {}/{}".format(ep_count,eval_times))
                
            for i in range(env.num_envs):
                total_reward.extend(episode_rewards[i])
                

            
            
            # filter the outliers
            total_reward.sort()
            total_reward = total_reward[10:-10]

            info = OrderedDict()
            info["ave_score"] = round(np.mean(total_reward), 1)
            info["ave_score_std"] = round(np.std(np.array(total_reward)),3)
            info["ave_action_reward"] = sum(total_reward)/total_actions_count
            if self.log_action_skill:
                info.update(action_statistic)
        else:
            # evaluate with single process env
            info = OrderedDict()
            if self.log_action_skill:
                action_statistic = OrderedDict()
                for i in range(env.action_space.n):
                    action_statistic[str(env.action_space[i])]=0
            ep_reward = []
            ep_ave_reward = []
            print("start to eval agent...")
            for i in range(eval_times):
                obs = env.reset()
                total_reward = []
                for i in range(eval_max_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, info_ = env.step(action)
                    total_reward.append(rewards[0])

                    if self.log_action_skill is True:
                        action_statistic[str(env.action_space[action[0]])] = action_statistic[str(env.action_space[action[0]])] + 1

                    if bool(dones[0]) is True:
                        break
                
                ep_reward.append(sum(total_reward))
                ep_ave_reward.append(sum(total_reward)/len(total_reward))
            
            
            print("Finish eval agent")
            print("Elapsed: {} sec".format(round(time.time()-self.strat_time, 3)))
            
            #filter the outliers
            ep_reward.sort()
            ep_reward = ep_reward[10:-10]

            ave_score = sum(ep_reward)/len(ep_reward)
            ave_action_reward = sum(ep_ave_reward)/len(ep_ave_reward)
            ave_score_std = round(np.std(np.array(ep_reward)),3)

            # info.update({"ave_score":ave_score, "ave_score_std":ave_score_std, "ave_reward":ave_reward})
            info["ave_score"] = ave_score
            info["ave_score_std"] = ave_score_std
            info["ave_action_reward"] = ave_action_reward
            if self.log_action_skill:
                info.update(action_statistic)
        return info
    def eval_callback(self, env, freq=1000, eval_times=100, eval_max_steps=int(1e6), save_path="./"):
        # self.n_steps = 0
        def callback(_locals, _globals):
            """
            Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
            :param _locals: (dict)
            :param _globals: (dict)
            """
            appro_freq = 0
            if freq < _locals["self"].n_batch:
                appro_freq = freq
            else:
                appro_freq = int(_locals["self"].n_batch*round(freq/_locals["self"].n_batch))
            
            eval_now = _locals.get("eval_now", False)
            if (_locals["self"].num_timesteps % appro_freq) == 0 or eval_now:
                print("start to eval at {} timesteps".format(_locals["self"].num_timesteps))
                start = datetime.datetime.now()
               
                obs = env.reset()
                ep_count = 0
                ep_rew = [0.0 for _ in range(env.num_envs)]
                episode_rewards = []
                for j in range(eval_max_steps):
                    actions, _states = _locals['self'].predict(obs)
                    obs, rewards, dones, info = env.step(actions)
                    for i in range(env.num_envs):
                        ep_rew [i] = ep_rew[i] + rewards[i]
                        if dones[i]:
                            
                            episode_rewards.append(ep_rew[i])
                            ep_rew[i] = 0
                            ep_count = ep_count + 1
                    if ep_count >= eval_times:
                        break
                episode_rewards.sort()
                episode_rewards = episode_rewards[10:-10]
                ave_score = round(np.mean(episode_rewards), 1)
                
                with open(os.path.join(save_path, "evaluate_score.txt"), 'a') as f:
                    print("{} {}".format(ave_score, _locals["self"].num_timesteps), file=f)
                print("Finish eval within {}".format(datetime.datetime.now()-start))
            
            return True
        
        
        return callback 
        
    def get_rewards(self, skills=[], train_total_timesteps=5000000, eval_times=100, eval_max_steps=int(1e6), model_save_name=None, add_info={}):
    

        """
        
        :param skills: (list) the availiable action sequence for agent 
        e.g [[0,2,2],[0,1,1]]
        :param train_total_timesteps: (int)total_timesteps to train 
        :param eval_times: (int)the evaluation times
        e.g eval_times=100, evalulate the policy by averageing the reward of 100 episode
        :param eval_max_steps: (int)maximum timesteps per episode when evaluate
        (deprecate):param model_save_name: (str)specify the name of saved model (should not repeat)
        :param add_info: (dict) other information to log in log.txt
        """

        
        if self.save_tensorboard and self.save_path is not None:
            tensorboard_log = os.path.join(self.save_path, "model_" + str(self._serial_num))
        else:
            tensorboard_log = None

        

        env_creator = lambda env:SkillWrapper(self.env_creator(env), skills=skills, gamma=self.gamma)
        
        
        if self.save_monitor is True:
            monitor_path = os.path.join(self.save_path, "monitor")
            try:
                os.makedirs(monitor_path)
            except OSError as ex:
                if ex.errno == errno.EEXIST and os.path.exists(monitor_path):
                    print("{} exists. ignore".format(monitor_path))
                    pass
                else:
                    raise 
        else:
            monitor_path=None



        
        if "cfg" in self.env_id:
            
            env = make_doom_env(self.env_id, self.num_cpu, self.seed, extra_wrapper_func=env_creator, logdir=monitor_path)

        else:
            env = VecFrameStack(make_atari_env(self.env_id, self.num_cpu, self.seed, extra_wrapper_func=env_creator, logdir=monitor_path), 4)


        model = None
        if self.use_converge_parameter is True:
            model = self.model(self.policy, env, verbose=self.verbose, tensorboard_log=tensorboard_log, n_steps=128, nminibatches=4,
                lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
                learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1)
        else:
            model = self.model(self.policy, env, verbose=self.verbose, tensorboard_log=tensorboard_log)

        self.strat_time = time.time()
        print("start to train agent...")

        callback=None 
        if self.evaluate_freq is not None and self.evaluate_freq>0:
            preiod_eval_path = os.path.join(self.save_path, "period_eval")
            mkdirs(preiod_eval_path)
            if "cfg" in self.env_id:
                
                eval_env = make_doom_env(self.env_id, self.num_cpu, self.seed, extra_wrapper_func=env_creator, logdir=monitor_path, wrapper_kwargs={"episode_life":False, "clip_rewards":False})
            else:
                eval_env = VecFrameStack(make_atari_env(self.env_id, self.num_cpu, self.seed, extra_wrapper_func=env_creator, logdir=preiod_eval_path, wrapper_kwargs={"episode_life":False, "clip_rewards":False}), 4)
            callback=self.eval_callback(eval_env, freq=self.evaluate_freq, eval_times=eval_times, eval_max_steps=eval_max_steps, save_path=preiod_eval_path)
        
        model.learn(total_timesteps=train_total_timesteps, reset_num_timesteps=self.reset_num_timesteps, callback=callback)
        print("Finish train agent")

        #evaluate once more because sometimes it is not divisible
        if callback is not None:
            callback({"self":model, "eval_now":True}, None)

        if self.save_path is not None:
            if self.preserve_model>0:
                
                self.save_model(model, skills=skills)

        

        env.close()
        # evaluate
        env = VecFrameStack(make_atari_env(self.env_id, self.num_cpu, self.seed, extra_wrapper_func=env_creator, logdir=None), 4)
        info = self.evaluate(env, model, eval_times, eval_max_steps)
        try:
            env.close() 
        except AttributeError as e:   
            print("Ignore : {}".format(e))
        try:
            del model
        except AttributeError as e:   
            print("Ignore del model : {}".format(e))
        
        
        #log result
        info.update(add_info)
        self.log(info)

        self._serial_num = self._serial_num + 1
        return info["ave_score"], info["ave_action_reward"]
    
    def save_model(self, model, **kwargs):
        
        name = "model_" + str(self._serial_num)
        sub_dir_name = os.path.join(self.save_path, "model_" + str(self._serial_num))
        mkdirs(sub_dir_name)

        save_name = os.path.join(sub_dir_name, name)
        if os.path.isfile(save_name+".pkl"):
            print("Warning: overwrite model: {}".format(save_name+".pkl"))
        model.save(save_name)

        if len(kwargs)!=0:
            with open('{}.yml'.format(save_name), 'w') as outfile:
                yaml.dump(kwargs, outfile, default_flow_style=False)

        
        if sub_dir_name not in self._save_model_name:
            self._save_model_name.append(sub_dir_name)

        if self._serial_num > self.preserve_model and self.preserve_model>0:
            remove_name = os.path.join(self.save_path, "model_" + str(self._serial_num-self.preserve_model))
            
            try:
                if os.path.exists(remove_name):
                    shutil.rmtree(remove_name)
            except OSError as ex:
                if errno.EACCES == ex.errno:
                    print("[Warn]: Can not remove ({})".format(remove_name))
                else:
                    raise
            
        
    def log(self, info=None):
        if info is not None and self.save_path is not None:
            filename = os.path.join(self.save_path, "log.txt")
            
            with open(filename, 'a') as f:
                
                print("Episode: {}".format(str(self._serial_num)), file=f)
                keys = info.keys()
                
                for key in keys:
                    print("{}: {}".format(key, info[key]), file=f)
                print("{s:{c}^{n}}".format(s="", c='*', n=27), file=f)




