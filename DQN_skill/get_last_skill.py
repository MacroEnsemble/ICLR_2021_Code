import numpy as np
import tensorflow as tf
import os
import sys
import argparse

from utils import get_latest_ckpt
from estimator import Estimator
from skill_env import AtariSkillEnv, TestEnv

parser = argparse.ArgumentParser()
parser.add_argument("game", type=str, help="Specify the atari game environment.")
parser.add_argument("version", type=str, help="Specify the version.")
args = parser.parse_args()

assert args.game in ["Alien", "Seaquest", "BeamRider", "Breakout",\
                     "SpaceInvaders", "Qbert", "Pong", "Enduro"], "Invalid environment!"

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

LEARNING_RATE = constant["LEARNING_RATE"]
NUM_SKILL = constant["NUM_SKILL"]
MAXLEN_PER_SKILL = constant["MAXLEN_PER_SKILL"]
LEN_SKILL = NUM_SKILL * MAXLEN_PER_SKILL        # Total Length of skills e.g (2,3)=>6, (1,9)=>9
VALID_ACTIONS = constant["VALID_ACTIONS"]       # -1 for degenerate
REWARD_FACTOR = constant["REWARD_FACTOR"]

checkpoint_dir = '../log/controller/{env}/dqn/{env}NoFrameskip_macro[3,3]_r{r}_v{version}_dqn/checkpoints/'.format(
                    env=args.game, r=REWARD_FACTOR, version=args.version)

env = AtariSkillEnv(env="{}NoFrameskip-v4".format(args.game), num_skill=NUM_SKILL,
                maxlen_per_skill=MAXLEN_PER_SKILL, action_space=len(VALID_ACTIONS), VALID_ACTIONS=VALID_ACTIONS,
                # save_path=CONTROLLER_SAVE_DIR,
                # spec_name=SPEC_NAME, config_path=CONFIG_PATH, worker_log_dir=WORKER_SAVE_DIR,
                cluster_timeout=8*60*60)

tf.reset_default_graph()

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


# Where we save our checkpoints and graphs
# experiment_dir = os.path.abspath("./{}/{}".format(CONTROLLER_SAVE_DIR, env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", lr=LEARNING_RATE, LEN_SKILL=LEN_SKILL, VALID_ACTIONS=VALID_ACTIONS)
target_estimator = Estimator(scope="target_q", lr=LEARNING_RATE, LEN_SKILL=LEN_SKILL, VALID_ACTIONS=VALID_ACTIONS)

# Saver to save/restore weights
saver = tf.train.Saver()
checkpoint_dir = '../dummy_log_test/checkpoints'

with tf.Session() as sess:
    ckpt_idx, ckpt = get_latest_ckpt(os.listdir(checkpoint_dir))
    if ckpt is not None:
        sys.path.append(checkpoint_dir)
        # ckpt = os.path.join(checkpoint_dir, ckpt)
        print('Restore model_{}.ckpt'.format(ckpt_idx))

        print('checkpoint_dir:', checkpoint_dir)
        print('-------------RESTORE WEIGHTS----------------')
        saver.restore(sess, '../dummy_log_test/checkpoints/model_950.ckpt')
        
        
        actions = []
        epsilon = 0   # Epsilon for this macro ensemble
        state = env.reset()

        policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

        while True:
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, _, done, _ = env.step(VALID_ACTIONS[action])
            # collect action sequence
            actions.append(VALID_ACTIONS[action])
            if done:
                break
            else:
                state = next_state

            # map action to workers

        print("Waiting for mapping...")
        print(actions)

