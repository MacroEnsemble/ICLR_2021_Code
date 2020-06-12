import numpy as np
from collections import namedtuple
from skill_env import MyEnv

Skill = namedtuple("Skill", ["id", "epsilon", "skill", "reward"])

def write_reward_to_log(macro_path, actions, reward, epsilon, NUM_SKILL, MAXLEN_PER_SKILL):
    """
    Args
        macro_path:
            string
            log path
        
        actions:
            VALID_ACTION, list
            actions be chosen.
        
        reward:
            float

        epsilon:
            
    Return
        None
    """
    skills = np.array(actions).reshape((NUM_SKILL, MAXLEN_PER_SKILL)).tolist()
    macro_log = open(macro_path, "a")
    macro_log.write("skills:{}, reward:{}, epsilon:{}\n".format(skills, reward, epsilon))
    macro_log.close()
    


def load_log(path):
    """
    Read macro.txt and return all the information
    for resume training process.
    
    Args:
        path: path to 'macro.txt'. string
    
    Retruns
        macro_log: [[generation, skills, reward], [generation, skills, reward], ...]. list
    """

    def decode_skills(line):
        assert "skills:" in line
        assert "reward:" in line
        # line[line.find("skill:")+len("skill:"):line.find("")]
        skills = line[line.find("["):line.rfind("]")+1]
        skills = str_to_skills(skills)

        if "epsilon" in line:
            reward = line[line.find("reward:")+len("reward:"):line.find("epsilon:")-2]
            epsilon = line[line.find("epsilon:")+len("epsilon:"):-1]
        else:
            reward = line[line.rfind(":")+1:-1]
            epsilon = 1
        
        if reward == "None":
            reward = None
        else:
            reward = float(reward)
        return skills, reward, epsilon

    macro_log = []
    i = 0
    with open(path, "r") as f:
        for line in f: 
            if line[:6] == "skills":
                skills, reward, epsilon = decode_skills(line)
                if reward is not None:
                    macro_log.append([skills, reward, epsilon])

    return macro_log

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

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

def macro_log_to_replay_memory(replay_memory, macro_log, VALID_ACTIONS, REWARD_FACTOR):
    '''
    Add macro_log into replay_memory 
    
    Args:
        replay_memory: list of Transition.
            Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        
        macro_log: list. [skills, rewards, log]
    
    Returns:
        replay_memory: Same as args.

    '''
    skills, reward, epsilon = macro_log

    # create Myenv which helps us generate states
    num_skill = len(skills)
    assert type(skills[0]) == list
    maxlen_per_skill = len(skills[0])
    action_space = len(VALID_ACTIONS)
    env = MyEnv(num_skill=num_skill,
                maxlen_per_skill=maxlen_per_skill,
                action_space=action_space,
                VALID_ACTIONS=VALID_ACTIONS)

    # map VALID_ACTIONS into actions && flatten skills
    skills_= []
    for skill in skills:
        for action in skill:
            action_ = VALID_ACTIONS.index(action)
            skills_.append(action_)
    skills = skills_
    del action_

    # Generate Transition for each action
    state = env.reset()

    # assert False
    states, next_states, actions, dones, rewards = [], [], [], [], []

    for action in skills:
        next_state, _, done, _ = env.step(VALID_ACTIONS[action])

        # Add data to cache
        states.append(state)
        next_states.append(next_state)
        actions.append(action)
        dones.append(done)
        
        if done:
            state = env.reset()
            rewards.append(reward)
        else:
            state = next_state
            rewards.append(0)

    for s, a, r, s_n, d in zip(states, actions, rewards, next_states, dones):
        if r is None: # the cluster worker failed
            continue
        f_r = r * REWARD_FACTOR
        replay_memory.append(Transition(s, a, f_r, s_n, d))

    return replay_memory


if __name__ == "__main__":
    log = load_log("./macro.txt")
    print(log)

    # rp = macro_log_to_replay_memory([], log, [-1,0,1,2,3,4,5], REWARD_FACTOR=0.01)

    # for r in rp:
    #     print(r)

    