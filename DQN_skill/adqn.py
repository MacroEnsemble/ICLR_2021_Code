import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import time
import sys
import tensorflow as tf

from lib import plotting
from collections import deque, namedtuple
from stable_baselines.common.policies import MlpPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2, TRPO, A2C, ACKTR, ACER
import argparse

from skill_env import AtariSkillEnv, TestEnv
from asynchronous_write_reward import write_reward_to_log, load_log, macro_log_to_replay_memory
from utils import read_score,  get_latest_ckpt
from train_agent import train_agent
from estimator import Estimator


parser = argparse.ArgumentParser()
parser.add_argument("game", type=str, help="Specify the atari game environment.")
parser.add_argument("version", type=str, help="Specify the version.")
parser.add_argument("--test", action="store_true", help="environment returns rewards really really fast.")
parser.add_argument("-d", "--dup", action="store_true", help="store in test logdir")
parser.add_argument("-s", "--server", type=str, default='s', 
                    help="Choose config. ex: master:hellokitty, worker:Nala. Then '-s hn'")
parser.add_argument("--rl_model", type=str, default="ppo", help="RL model.")
# parser.add_argument("--doom", action="store_true", help="activate doom environment.")
args = parser.parse_args()

assert args.game in ["Alien", "Seaquest", "BeamRider", "Breakout",\
                     "SpaceInvaders", "Qbert", "Pong", "Enduro", "KungFuMaster", 
                     "MsPacman", "Phoenix", "Defender", "Amidar", "Atlantis", "Asteroids", 
                     "Gravitar", "Frostbite", "Solaris", "CrazyClimber", "Riverraid", 
                     "PrivateEye", "Hero", "health_gathering", "Zaxxon", "Boxing"], "Invalid environment!"

# dirty code
if args.game == "Alien":
    from constant import Alien_constant as constant
elif args.game == "Seaquest":
    from constant import Seaquest_constant as constant
elif args.game == "BeamRider":
    from constant import BeamRider_constant as constant
elif args.game == "Breakout":
    from constant import Breakout_constant as constant
elif args.game == "SpaceInvaders":
    from constant import SpaceInvaders_constant as constant
elif args.game == "Qbert":
    from constant import Qbert_constant as constant
elif args.game == "Pong":
    from constant import Pong_constant as constant
elif args.game == "Enduro":
    from constant import Enduro_constant as constant
elif args.game == "KungFuMaster":
    from constant import KungFuMaster_constant as constant
elif args.game == "MsPacman":
    from constant import MsPacman_constant as constant
elif args.game == "Phoenix":
    from constant import Phoenix_constant as constant
elif args.game == "Defender":
    from constant import Defender_constant as constant
elif args.game == "Amidar":
    from constant import Amidar_constant as constant
elif args.game == "Atlantis":
    from constant import Atlantis_constant as constant
elif args.game == "Asteroids":
    from constant import Asteroids_constant as constant
elif args.game == "Gravitar":
    from constant import Gravitar_constant as constant
elif args.game == "Frostbite":
    from constant import Frostbite_constant as constant
elif args.game == "Solaris":
    from constant import Solaris_constant as constant
elif args.game == "CrazyClimber":
    from constant import CrazyClimber_constant as constant
elif args.game == "Riverraid":
    from constant import Riverraid_constant as constant
elif args.game == "Boxing":
    from constant import Boxing_constant as constant
elif args.game == "PrivateEye":
    from constant import PrivateEye_constant as constant
elif args.game == "Hero":
    from constant import Hero_constant as constant
elif args.game == "health_gathering":
    from constant import health_gathering as constant
elif args.game=="Zaxxon":
    from constant import Zaxxon_constant as constant
elif args.game=="Boxing":
    from constant import Boxing_constant as constant
else:
    raise NotImplementedError

# Arguments of experiment
VALID_ACTIONS = constant["VALID_ACTIONS"]       # -1 for degenerate
NUM_SKILL = constant["NUM_SKILL"]
MAXLEN_PER_SKILL = constant["MAXLEN_PER_SKILL"]
LEN_SKILL = NUM_SKILL * MAXLEN_PER_SKILL        # Total Length of skills e.g (2,3)=>6, (1,9)=>9
TOTAL_NUM_OF_WORKER = constant["TOTAL_NUM_OF_WORKER"]
UPDATE_FREQ = LEN_SKILL * TOTAL_NUM_OF_WORKER   # Total Length of skills training at the same time e.g 20 woker with macro shape(2,3)=> 20*2*3=120

# Hyperparameters
REWARD_FACTOR = constant["REWARD_FACTOR"]
NUM_EPOCH = constant["NUM_EPOCH"]
LEARNING_RATE = constant["LEARNING_RATE"]
PENALTY = constant["PENALTY"]
BATCH_SIZE = constant["BATCH_SIZE"]
NUM_INIT_SKILL = constant["NUM_INIT_SKILL"]
MAX_NUM_TOTAL_SKILL = constant["MAX_NUM_TOTAL_SKILL"]
MAX_LIMIT_TIME = constant["MAX_LIMIT_TIME"]
NUM_EXPLORE_SKILL = constant["NUM_EXPLORE_SKILL"]

NUM_EXPLORE_GENERATION = (NUM_EXPLORE_SKILL - NUM_INIT_SKILL) // TOTAL_NUM_OF_WORKER + 1
NUM_END_GENERATION = (MAX_NUM_TOTAL_SKILL - NUM_INIT_SKILL) // TOTAL_NUM_OF_WORKER + 1

# Saving directories
GLOBAL_LOG_DIR="../log" # all log will saved here
if args.test:
    GLOBAL_LOG_DIR = "../dummy_log_test"

CONTROLLER_SAVE_DIR =  os.path.join(GLOBAL_LOG_DIR, "controller", args.game, "dqn") # controller will saved here
WORKER_SAVE_DIR = os.path.join(GLOBAL_LOG_DIR, "worker", args.game, "dqn") # worker will saved here

if args.rl_model == 'ppo':
    CONTROLLER_SAVE_DIR = os.path.join(CONTROLLER_SAVE_DIR, 'ppo')
    WORKER_SAVE_DIR = os.path.join(WORKER_SAVE_DIR, 'ppo')
elif args.rl_model == 'a2c':
    CONTROLLER_SAVE_DIR = os.path.join(CONTROLLER_SAVE_DIR, 'a2c')
    WORKER_SAVE_DIR = os.path.join(WORKER_SAVE_DIR, 'a2c')
else:
    raise NotImplementedError


# worker related parameter
SPEC_NAME = "{}NoFrameskip_macro[{},{}]_r{}_v{}_dqn".format(
                        args.game ,NUM_SKILL, MAXLEN_PER_SKILL, REWARD_FACTOR, args.version) # experiment name
if args.dup:
    SPEC_NAME = SPEC_NAME + "_dup"
if args.test:
    SPEC_NAME = SPEC_NAME + "_test"

print("SPEC_NAME:", SPEC_NAME)

if args.server is not None:
    CONFIG_PATH = "../grpc_cluster/cluster_config_dqn_{}.yml".format(args.server)
else:
    CONFIG_PATH = "../grpc_cluster/cluster_config_dqn.yml"

print("Building environment")
# if args.test:
#     VALID_ACTIONS = [0,1,2,3,4]
#     env = TestEnv(spec_name=SPEC_NAME, env="{}NoFrameskip-v4".format(args.game), num_skill=NUM_SKILL, maxlen_per_skill=MAXLEN_PER_SKILL,
#                 action_space=len(VALID_ACTIONS), VALID_ACTIONS=VALID_ACTIONS,
#                 save_path = CONTROLLER_SAVE_DIR, verbose=1)
# else:
env = AtariSkillEnv(env="{}NoFrameskip-v4".format(args.game), num_skill=NUM_SKILL,
                maxlen_per_skill=MAXLEN_PER_SKILL, action_space=len(VALID_ACTIONS), VALID_ACTIONS=VALID_ACTIONS,
                save_path=CONTROLLER_SAVE_DIR,
                spec_name=SPEC_NAME, config_path=CONFIG_PATH, worker_log_dir=WORKER_SAVE_DIR,
                cluster_timeout=8*60*60, server_names=args.server, TEST=args.test, model=args.rl_model)
print("Env built")

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def check_duplicated_macro_action(replay_memory, states, actions, next_states, dones):
    '''
    1. Add penalty and terminal state to duplicated macro actions.
    2. Add duplicated macro actions transitions into replay memory.
    3. Remove duplicated macro actions from cache.
    ''' 

    assert len(actions) % LEN_SKILL == 0
    assert dones[-1] == True
    
    # get macro action
    ma_states = states[-LEN_SKILL:]     # ma for macro action
    ma_next_states = next_states[-LEN_SKILL:]
    ma_actions = actions[-LEN_SKILL:]
    ma_dones = dones[-LEN_SKILL:]

    # compare whether ma_actions contains duplicated macro action
    macro_actions = [ ma_actions[i*MAXLEN_PER_SKILL:(i+1)*MAXLEN_PER_SKILL] for i in range(NUM_SKILL)]

    # compare macro actions to each others
    arg_dup_macro = None
    for i in range(1, len(macro_actions)):
        for j in range(i):
            if macro_actions[i] == macro_actions[j]:
                arg_dup_macro = i       # The ith macro action is duplicated
                break
        if arg_dup_macro is not None:
            break

    if arg_dup_macro is None:
        # print("Send macro_actions:", macro_actions)
        return replay_memory, states, next_states, actions, dones, False  # Modify nothing

    else:
        # print("Send macro_actions:", macro_actions[:arg_dup_macro-1])
        # remove transitions after duplicated macro actions
        ma_states = ma_states[:(arg_dup_macro+1)*MAXLEN_PER_SKILL]
        ma_actions = ma_actions[:(arg_dup_macro+1)*MAXLEN_PER_SKILL]
        ma_next_states = ma_next_states[:(arg_dup_macro+1)*MAXLEN_PER_SKILL]
        ma_dones = ma_dones[:(arg_dup_macro+1)*MAXLEN_PER_SKILL]
                
        # Add penalty and terminal state to duplicated macro actions.
        ma_rewards = [0 for i in range(len(ma_states))]
        ma_rewards[-1] = PENALTY
        ma_dones[-1] = True

        # Add duplicated macro actions transitions into replay memory
        for s, a, r, s_n, d in zip(ma_states, ma_actions, ma_rewards, ma_next_states, ma_dones):
            f_r = r * REWARD_FACTOR
            replay_memory.append(Transition(s, a, f_r, s_n, d))

        # Remove duplicated macro actions from the cache
        return replay_memory, states[:-LEN_SKILL], next_states[:-LEN_SKILL], actions[:-LEN_SKILL], dones[:-LEN_SKILL], True


print("Start DQN")
        
# Main algorithm
def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    # checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    macro_dir = os.path.join(experiment_dir, "macro")
    macro_path = os.path.join(macro_dir, "macro.txt")

    worker_dir = os.path.abspath("./{}/{}".format(WORKER_SAVE_DIR, env.spec.id))
    output_path = os.path.join(worker_dir, "output_score")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    if not os.path.exists(macro_dir):
        os.makedirs(macro_dir)
    
    # Saver to save/restore weights
    saver = tf.train.Saver()
    
    ckpt_idx, ckpt = get_latest_ckpt(os.listdir(checkpoint_dir))
    if ckpt is not None:
        os.system('cp -r {} ./tmp/'.format(checkpoint_dir))
        ckpt = os.path.join('./tmp/checkpoints', ckpt)
        
        saver.restore(sess, ckpt)
        print('Restore model_{}.ckpt'.format(ckpt_idx))

    # The replay memory
    replay_memory = []

    total_t = sess.run(tf.contrib.framework.get_global_step())  # total_t = 0, for q value
    tf_update_idx = 0   # for plotting graph (reward and epsilon)

    # Restore replay memory if exists
    if os.path.isfile(macro_path):
        macro_log = load_log(macro_path)    # [[skills, reward, epsilon], [skills, reward, epsilon], ...]
        print("Reading macro.txt")
        print("Resume training from {} skills...".format(len(macro_log)))
        for log_skills in macro_log:
            replay_memory = macro_log_to_replay_memory(
                                        replay_memory, log_skills, VALID_ACTIONS, REWARD_FACTOR)
            tf_update_idx += 1      # for plotting

    assert tf_update_idx >= ckpt_idx, 'Unexpected checkpoint. Checkpoint version is higher than the replay memory.'

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, NUM_EXPLORE_SKILL)
    _ = np.full(shape=(MAX_NUM_TOTAL_SKILL-NUM_EXPLORE_SKILL,), fill_value=epsilon_end)
    epsilons = np.concatenate((epsilons, _), axis=0)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))
    
    state = env.reset()

    strat_time = time.time()

    # populating initial skills, random generate skill
    while tf_update_idx < NUM_INIT_SKILL:
        # Generate action sequence
        print("{}/{} skills @ Init replay memory".format(tf_update_idx+1, NUM_INIT_SKILL))
        actions = []
        state = env.reset()
        epsilon = epsilons[0]        # 1.0
        while True:
            action_probs = policy(sess, state, epsilon)     # epsilon = 1.0, totally random search
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, _, done, _ = env.step(VALID_ACTIONS[action])
            # collect actions sequence
            actions.append(VALID_ACTIONS[action])
            if done:
                break

            else:
                state = next_state

        # map action to workers
        print("Waiting for mapping...")
        env.map_single_reward(actions, epsilon)      # return when actions is sent to a worker
        
        # check if any macro ensemble be calculated.
        while tf_update_idx < NUM_INIT_SKILL:
            skill, reward, epsilon = read_score(output_path)    # Reutrn [None, None] if nothing new
            
            if skill is not None:
                factor_episode_reward = reward * REWARD_FACTOR
                # Write log into tensorboard
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=factor_episode_reward, node_name="factor_episode_reward", tag="factor_episode_reward")
                episode_summary.value.add(simple_value=reward, node_name="episode_reward", tag="episode_reward")
                episode_summary.value.add(simple_value=epsilon, node_name="epsilon", tag="epsilon")
                episode_summary.value.add(simple_value=LEN_SKILL, node_name="episode_length", tag="episode_length")
                q_estimator.summary_writer.add_summary(episode_summary, tf_update_idx)
                q_estimator.summary_writer.flush()
                tf_update_idx += 1
                write_reward_to_log(macro_path, skill, reward, epsilon,
                                NUM_SKILL=NUM_SKILL, MAXLEN_PER_SKILL=MAXLEN_PER_SKILL)
                    
                # add the new macro ensemble into the replay buffer
                skill = np.array(skill).reshape(-1).tolist()
                
                state = env.reset()
                for a in skill:
                    next_state, _, done, _ = env.step(a)    # a has been mapped to VALID_ACTIONS

                    # Add data to replay memory
                    if done:
                        replay_memory.append(Transition(state, VALID_ACTIONS.index(a), factor_episode_reward, next_state, done))
                        
                    else:
                        replay_memory.append(Transition(state, VALID_ACTIONS.index(a), 0, next_state, done))
                        state = next_state
            else:
                break

    # train model (Number of loaded macro > NUM_INIT_MODEL)
    if tf_update_idx > NUM_INIT_SKILL:
        for i in range(tf_update_idx - NUM_INIT_SKILL):
            partial_memory = replay_memory[:LEN_SKILL*(NUM_INIT_SKILL + i)]
            # Train agent
            if NUM_INIT_SKILL+i > ckpt_idx:
                q_estimator, target_estimator = train_agent(
                    sess=sess,
                    replay_memory=partial_memory,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    NUM_EPOCH=NUM_EPOCH,
                    discount_factor=discount_factor,
                    BATCH_SIZE=BATCH_SIZE,
                    NUM_TOTAL_SKILL=MAX_NUM_TOTAL_SKILL,
                    update_target_estimator_every=update_target_estimator_every,
                    tf_update_idx=NUM_INIT_SKILL+i,        # current skill id
                )
            

    # Main training loop
    skill = []       # Data sent back from worker. Initialize skill to null.
    while tf_update_idx < MAX_NUM_TOTAL_SKILL:
        if time.time()-strat_time > MAX_LIMIT_TIME:
            break
        # Save model
        if tf_update_idx % 50 == 0:
            ckpt = "model_{}.ckpt".format(tf_update_idx)
            print('Saving model_{}.ckpt'.format(tf_update_idx))
            saver.save(sess, os.path.join(checkpoint_dir, ckpt))
            

        # If our replay memory is full, pop the first element
        while len(replay_memory) > replay_memory_size:
            replay_memory.pop(0)

        if skill is not None:
            # Train agent with
            if tf_update_idx > ckpt_idx:
                
                q_estimator, target_estimator = train_agent(
                    sess=sess,
                    replay_memory=replay_memory,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    NUM_EPOCH=NUM_EPOCH,
                    discount_factor=discount_factor,
                    BATCH_SIZE=BATCH_SIZE,
                    NUM_TOTAL_SKILL=MAX_NUM_TOTAL_SKILL,
                    update_target_estimator_every=update_target_estimator_every,
                    tf_update_idx=tf_update_idx,
                )
        
        # populate a new macro ensemble
        if skill is None:
            print("{}/{} th skills".format(tf_update_idx+1, MAX_NUM_TOTAL_SKILL))
            actions = []
            epsilon = epsilons[tf_update_idx-NUM_INIT_SKILL]   # Epsilon for this macro ensemble
            state = env.reset()
            while True:
                action_probs = policy(sess, state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, _, done, _ = env.step(VALID_ACTIONS[action])
                # collect action sequence
                actions.append(VALID_ACTIONS[action])
                if done:
                    break
                    # if args.dup:    # duplicated macro action in one ensemble is valid.
                    #     break
                    # else:           # add penalty to duplicated macro action
                    #     replay_memory, states, next_states, actions, dones, is_dup = check_duplicated_macro_action(
                    #                                                         replay_memory=replay_memory,
                    #                                                         states=states, next_states=next_states,
                    #                                                         actions=actions, dones=dones)
                    #     if is_dup:
                    #         continue
                    #     else:
                    #         break
                else:
                    state = next_state

            # map action to workers
            print("Waiting for mapping...")
            env.map_single_reward(actions, epsilon)      # return when actions is sent to a worker


        # check if any macro ensemble be calculated.    
        skill, reward, epsilon = read_score(output_path)    # Reutrn [None, None] if nothing new
        
        if skill is not None:
            factor_episode_reward = reward * REWARD_FACTOR
            # Write log into tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=factor_episode_reward, node_name="factor_episode_reward", tag="factor_episode_reward")
            episode_summary.value.add(simple_value=reward, node_name="episode_reward", tag="episode_reward")
            episode_summary.value.add(simple_value=epsilon, node_name="epsilon", tag="epsilon")
            episode_summary.value.add(simple_value=LEN_SKILL, node_name="episode_length", tag="episode_length")
            q_estimator.summary_writer.add_summary(episode_summary, tf_update_idx)
            q_estimator.summary_writer.flush()
            tf_update_idx += 1
            write_reward_to_log(macro_path, skill, reward, epsilon,
                            NUM_SKILL=NUM_SKILL, MAXLEN_PER_SKILL=MAXLEN_PER_SKILL)
                
            # add the new macro ensemble into the replay buffer
            skill = np.array(skill).reshape(-1).tolist()
            
            state = env.reset()
            for a in skill:
                next_state, _, done, _ = env.step(a)    # a has been mapped to VALID_ACTIONS

                # Add data to replay memory
                if done:
                    replay_memory.append(Transition(state, VALID_ACTIONS.index(a), factor_episode_reward, next_state, done))
                    
                else:
                    replay_memory.append(Transition(state, VALID_ACTIONS.index(a), 0, next_state, done))
                    state = next_state
            READ_REWARD_AGAIN = True

    env.close()
    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./{}/{}".format(CONTROLLER_SAVE_DIR, env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir, lr=LEARNING_RATE, LEN_SKILL=LEN_SKILL, VALID_ACTIONS=VALID_ACTIONS)
target_estimator = Estimator(scope="target_q", lr=LEARNING_RATE, LEN_SKILL=LEN_SKILL, VALID_ACTIONS=VALID_ACTIONS)

# gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
# gpu_config.gpu_options.allow_growth=True

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning(sess,
        env,
        q_estimator=q_estimator,
        target_estimator=target_estimator,
        experiment_dir=experiment_dir,
        num_episodes=2000,
        replay_memory_size=1e4,
        update_target_estimator_every=4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        discount_factor=0.99)


