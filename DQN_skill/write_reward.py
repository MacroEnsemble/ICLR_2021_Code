import numpy as np
from collections import namedtuple
from skill_env import MyEnv

def write_reward_to_log(macro_path, actions, rewards, generation,
                        UPDATE_FREQ, LEN_SKILL, NUM_SKILL, MAXLEN_PER_SKILL):
    """
    Args
        macro_path
            string
            log path
        
        actions
            list, len(actions) = UPDATE_FREQ
            actions been chose.
        
        rewards
            list. len(rewards) = UPDATE_FREQ
            rewards. includes None when cluster fails. 
            
        generation
            int
            Current generation. Be used to print log.

    Return
        None

    """

    macro_log = open(macro_path, "a")

    # Titile
    if generation == 0:
        macro_log.write("-----Random Initial Search-----\n")
    else:
        macro_log.write("-----Generation {}-----\n".format(generation))

    # Preprocess reward, eliminate 0 terms
    rewards_ = []
    for i in range(0, UPDATE_FREQ, LEN_SKILL):
        try:
            pass
            # assert rewards[i+LEN_SKILL-1] != 0
        except IndexError:
            print("-"*20)
            print("len(rewards):", len(rewards))
            print("rewards")
            print(rewards)
            exit(0)
        rewards_.append(rewards[i+LEN_SKILL-1])
    rewards = rewards_

    # Preprocess skills, change shape
    actions = np.array(actions).reshape(-1, NUM_SKILL, MAXLEN_PER_SKILL)
 
    # Preprocess skills&rewards, eliminate reward = None
    # rewards_, actions_ = [], []
    # for a, r in zip(actions, rewards):
    #     if r is None:
    #         continue
    #     actions_.append(a)
    #     rewards_.append(r)
    # actions = np.array(actions_)
    # rewards = rewards_

    for skills, r in zip(actions, rewards):
        if r is None:
            r = "None"
        macro_log.write("skills:{}, reward:{}\n".format(skills.tolist(), r))
    
    # Average reward
    rewards = [r for r in rewards if r is not None]     # eliminate None terms
    avg_score = np.sum(rewards) / len(rewards)
    macro_log.write("average reward: {}\n".format(avg_score))
  
def load_log(path):
    """
    Read macro.txt and return all the information
    for resume training process.
    
    Args:
        path: path to 'macro.txt'. string
    
    Retruns
        macro_log: [[generation, skills, reward], [generation, skills, reward], ...]. list
    """
    def decode_title(line):
        assert line[0] == "-"
        if line == "-----Random Initial Search-----\n":
            return 0    # Generation 0
        elif "Generation" in line:
            line = line.replace("\n", "")
            line = line.replace("-", "")
            line = line.replace("Generation ", "")
            generation = int(line)
            return generation

    def decode_skills(line):
        assert line[:6] == "skills"
        skills = line[line.find("["):line.rfind("]")+1]
        skills = str_to_skills(skills)
        reward = line[line.rfind(":")+1:-1]
        if reward == "None":
            reward = None
        else:
            reward = float(reward)
        return skills, reward

    macro_log = []
    i = 0
    with open(path, "r") as f:
        for line in f: 
            if line[0] == "-":
                generation = decode_title(line)
            if line[:6] == "skills":
                skills, reward = decode_skills(line)
                if reward is not None:
                    macro_log.append([generation, skills, reward])    

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
        
        macro_log: list. [generation, skills, rewards]
    
    Returns:
        replay_memory: Same as args.

    '''
    generation, skills, reward = macro_log

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
    log = log[0]
    rp = macro_log_to_replay_memory([], log, [-1,0,1,2,3,4,5], REWARD_FACTOR=0.01)

    # for r in rp:
    #     print(r)

    