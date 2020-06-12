import numpy as np
import sys, os
sys.path.append(os.path.abspath("../cluster"))
import sshex, sshex_c, sshex.logger

# from grpc_cluster import Cluster
import errno
import gym
# from skill_lib.env_wrapper import SkillWrapper, ActionRemapWrapper
# from skill_lib.manager import AtariPolicyManager
from stable_baselines.common.vec_env import DummyVecEnv

from generate_config_adqn import gen_config



class Spec:
    def __init__(self, name):
        self.id = name

class MyEnv:
    def __init__(self,
                spec_name="VanillaEnv",
                env=None,
                # Macro action spec
                num_skill=1,
                maxlen_per_skill=4,
                action_space=5,
                VALID_ACTIONS=[],
                # Path
                save_path="./path/to/store/location",
                worker_log_dir="../test_log"):

        self.env = env
        self.spec = Spec(spec_name)
        self.env_id = env

        self.num_skill = num_skill
        self.maxlen_per_skill = maxlen_per_skill
        self.len_skill = num_skill * maxlen_per_skill
        self.action_space = action_space
        self.VALID_ACTIONS = VALID_ACTIONS
        
        self.state = np.zeros((self.len_skill, action_space))    # observation
        self.ac_counter = 0 # Counter for actions choosed. When ac_counter equals to len_skill, game end and return a reward.
          
        self.save_path = save_path # controller log(tensorboard) save path
        self.worker_log_dir = worker_log_dir
        self.reset_counter = 0

    def reset(self):
        self.state = np.zeros((self.len_skill, self.action_space))
        self.ac_counter = 0       
        return self.state.copy()

    def step(self, action):
        '''
        Args
            action: int, vary from 0 to action space
        Return
            next_state: np.array, shape=(self.len_skill, self.action_space)
            reward:     int, basically zero, when complete a skill, get a final reward.
            done:       bool, False when not done. True when the whole skill is generated.
            _:          Whatever

        '''
        # remap by VALID_ACTIONS
        action = self.VALID_ACTIONS.index(action)

        # one hot encoding
        onehotaction = np.zeros(self.action_space)
        onehotaction[action] = 1

        # insert one-hot action into current action
        self.state[self.ac_counter] = onehotaction
        self.ac_counter += 1
        
        reward = 0

        if self.ac_counter == self.len_skill:
            done = True
        else:
            done = False

        return self.state.copy(), reward, done, None
    
    def close(self):
        pass

class AtariSkillEnv(MyEnv):
    def __init__(self,
                spec_name="VanillaEnv",
                env="Alien-ramDeterministic-v4",
                # Macro action spec
                num_skill=1,
                maxlen_per_skill=4,
                action_space=5,
                VALID_ACTIONS=[],     # remap nn output to action
                model=None,
                # Path
                config_path='../grpc_cluster/cluster_config.yml',
                save_path = "./path/to/store/location",
                worker_log_dir="../test_log",
                cluster_timeout=4*60*60,
                server_names = 'hjmnoa',
                TEST = False,
                ):

        super(AtariSkillEnv, self).__init__(spec_name=spec_name, env=env, num_skill=num_skill, 
                        maxlen_per_skill=maxlen_per_skill,
                        action_space=action_space, VALID_ACTIONS=VALID_ACTIONS,
                        save_path=save_path, worker_log_dir=worker_log_dir)
       
        # self.cluster = Cluster(config_path)
        # self.cluster_timeout = cluster_timeout
        self.model = model
        self.TEST = TEST
        print("[SUCEESS create cluster]")
        sshex.logger.Config.Use(filename='{}'.format(os.path.join(save_path, "error_log.txt")),
                  colored=True,
                  level='DEBUG')
        # === sshex config ===
        config = gen_config(server_names=server_names, env=self.env)
        sshex_c.Config(config)

    def map_single_reward(self, actions, info):
        ''' For asychronous dqn
        Args
            actions:
                list, len = len_skill
                single macro ensemble.

        Returns
            None
        '''

        # Restructure actions
        actions = np.array(actions).reshape((self.num_skill, self.maxlen_per_skill)).tolist()
        actions = str(actions).replace(" ", "")

        # $&^%$&^%$&^
        worker_log_dir = os.path.join(self.worker_log_dir, self.spec.id)

        if self.TEST:
            train_total_timesteps = 100
        else:
            train_total_timesteps = int(5e6)

        data = [['source ~/skill_search.sh; ',  'python3', 'atari_cmd.py',\
                '--env_id', self.env, '--skill', '{}'.format(actions),\
                '--logdir', worker_log_dir, '--info', info,\
                '--duplicate_checker', 'True', '--empty_action', '-1',\
                '--train_total_timesteps', train_total_timesteps,\
                '--rl_model', self.model,\
                '--save_monitor', "False"]]

        # self.cluster.map(data)
        sshex_c.map(data, wait=False, retry=True, cooldown=30)
        
        return None

    def get_reward_by_cache(self, actions):
        ''' For synchronous dqn
        Args
            actions:
                list, len = len_skill *  TOTAL_NUM_OF_WORKER
                Cache actions which RL agent chose.
        Returns
            rewards:
                list, len(reward)==len(action)
        '''
        assert len(actions) % self.len_skill == 0

        num = len(actions) // self.len_skill   # num is TOTAL_NUM_OF_WORKER

        # map action to VALID_ACTIONS
        actions_ = [self.VALID_ACTIONS[a] for a in actions]
        actions = actions_

        # actions to states
        train_states = []
        for i in range(num):
            skills = actions[i*self.len_skill:(i+1)*self.len_skill] # shape = (len_skill,)
            skills = np.array(skills)
            skills = np.reshape(skills, (self.num_skill, self.maxlen_per_skill)).tolist()   # shape = (num_skill, maxlen_per_skill)
            train_states.append(skills)
     
        # add cluster
        print("[SEND SKILLS]: {}".format(train_states))
        # worker_log_dir = os.path.join(self.worker_log_dir, self.spec.id+"_worker")
        worker_log_dir = os.path.join(self.worker_log_dir, self.spec.id)

        if self.TEST:
            train_total_timesteps = 50
        else:
            train_total_timesteps = int(5e6)

        data = [{"skills":train_states[i], "logdir":worker_log_dir, 'env_id': self.env_id, 'empty_action':-1,
                "train_total_timesteps":train_total_timesteps} for i in range(len(train_states))]

        assert self.cluster.map(data, timeout=self.cluster_timeout)     
        done, rewards = self.cluster.reduce(block=True)     # takes time
        print("-"*20)
        print("[rewards]")
        print("type:", type(rewards))
        print("value:")
        print(rewards)

        total_reward = []
        
        try:
            assert len(train_states) == len(rewards)
        except AssertionError:
            print("len(train_states): {}".format(len(train_states)))
            print("len(rewards): {}".format(len(rewards)))
            exit(0)

        # Zero padding to rewards
        for i in range(len(train_states)):
            if rewards[i] is not None:
                for j in range(self.len_skill-1):
                    total_reward.append(0)
                total_reward.append(rewards[i])
            else:
                # Sometimes cluster return None
                for j in range(self.len_skill-1):
                    total_reward.append(None)
                total_reward.append(rewards[i]) # None
        
        scores_dir = os.path.join(self.save_path, self.spec.id)
        try:
            os.makedirs(scores_dir)
        except OSError as ex:
            if ex.errno == errno.EEXIST and os.path.exists(scores_dir):
                print("[CRTICAL SECTION] create two same dir: {} create".format(scores_dir))
                pass
            else:
                raise 
                
        scores_file = os.path.join(scores_dir,"score.txt")
        with open(scores_file, 'a') as f:
            for s, r in zip(train_states, rewards):
                if r is not None:
                    print("{}:{}".format(s,r), file=f)
        
        assert len(total_reward) == len(actions)
        return total_reward
    


# from get_reward import RewardGenerator  # ground search data
class TestEnv(MyEnv):
    def __init__(self,
                spec_name="VanillaEnv",
                env="Alien-ramDeterministic-v4",
                # Macro action spec
                num_skill=1,
                maxlen_per_skill=4,
                action_space=5,
                VALID_ACTIONS=[],     # remap nn output to action
                # Path
                save_path = "./path/to/store/location",
                worker_log_dir="../test_log"):

        super(TestEnv, self).__init__(spec_name=spec_name, env=env, num_skill=num_skill, maxlen_per_skill=maxlen_per_skill,
                        action_space=action_space, VALID_ACTIONS=VALID_ACTIONS,
                        save_path=save_path, worker_log_dir=worker_log_dir)
       
        # self.RG = RewardGenerator()
        raise NotImplementedError
    
    def get_reward_by_cache(self, actions):
        '''
        Args
            actions:
                list, len > num Cache actions which RL agent chose.
        Return
            rewards:
                list, len(reward)==len(action)
        '''
        assert len(actions) % self.len_skill == 0
        assert self.maxlen_per_skill % 3 == 0

        num = len(actions) // self.len_skill   # num is TOTAL_NUM_OF_WORKER


        train_states = []
        rewards = []

        # actions to skills, list to nest list
        for i in range(num):
            skills = actions[i*self.len_skill:(i+1)*self.len_skill]
            train_states.append(skills)

        # Calculate rewards
        for skills in train_states:
            r = 0
            num_split = self.maxlen_per_skill // 3
            for i in range(0, self.maxlen_per_skill, 3):
                s = skills[i:i+3]
                r = r + self.RG.get_reward(str(s).replace(' ', ''))
                
            r = r / num_split
            rewards.append(r)

        total_reward = []

        for i in range(len(train_states)):
            for _ in range(self.len_skill-1):
                total_reward.append(0)
            total_reward.append(rewards[i])

        assert len(total_reward) == len(actions)
        
        return total_reward
    

if __name__ == '__main__':
    
    def test_env(EnvClass):
        env = EnvClass()
        state = env.reset()
        print(state)
        
        steps = [0,1,2]

        for step in steps:
            state, reward, done, _ = env.step(step)
            print(state)


    env_list = [AtariSkillEnv, TestEnv]
    for EnvClass in env_list:
        test_env(EnvClass)
    




