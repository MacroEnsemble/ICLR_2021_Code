all_exp= dict(
### ============ kungfumaster ============ ###

asteroids_dqn_ppo = [
    {
        "env_id":"Asteroids",
        "rl_model":"ppo",
        "subdir":"../log/other/macro_ensemble/Asteroids/dqn/ppo/ori",
        "seed":2000,
        "macro":[
            [],
        ]
    },
    {
        "env_id":"Asteroids",
        "rl_model":"ppo",
        "subdir":"../log/other/macro_ensemble/Asteroids/dqn/ppo/top",
        "seed":2000,
        "macro":[
            [[4, 3, 1], [1, 3], [1, 0, 0]],
        ]
    },

    {
        "env_id":"Asteroids",
        "rl_model":"ppo",
        "subdir":"../log/other/macro_ensemble/Asteroids/dqn/ppo/disassemble",
        "seed":2000,
        "macro":[
            [[4, 3, 1]], 
            [[1, 3]],
            [[1, 0, 0]]
        ]                  
    }
]
,asteroids_drl_with_macro_ppo = [
    {
        "env_id":"Asteroids",
        "rl_model":"ppo",
        "subdir":"../log/other/baselines/drl_with_macro/retrain/Asteroids/top",
        "seed":2000,
        "macro":[
            [[1, 1, 1], [7, 7, 1], [1, 7, 0]] ,  
        ]                  
    },
    {
        "env_id":"Asteroids",
        "rl_model":"ppo",
        "subdir":"../log/other/baselines/drl_with_macro/retrain/Asteroids/disassemble",
        "seed":2000,
        "macro":[
            [[1, 1, 1]], 
            [[7, 7, 1]], 
            [[1, 7, 0]],  
        ]                  
    }
]

)
import os, sys
from analyze_score import get_maro_info, sort_score
from utils import str_to_skills 
import re
sys.path.append(os.path.abspath("../DQN_skill"))
from constant import env_list
def serach_envID_from_path(path):
    searched = False
    # m = re.search("[A-Z][a-z]*-ramDeterministic-v4", model_path)
    env_id = None
    m = re.search("[A-Z][a-z]*NoFrameskip-v4", path)
    if m is not None:
        searched = True
        env_id = m.group(0)
        env_id.replace("NoFrameskip-v4", "")
    
    if searched is not True:
        for id_ in env_list:
            if  id_.lower() in path.lower():
                searched = True
                env_id = id_ 

    
    return env_id

def generate_exp_config(path, mode="top"):
    info = get_maro_info(path)
    if mode=="last":
        data = sort_score(info, sort_method="index")
        macro = str_to_skills(data[0][0])
    elif mode=="top":
        data = sort_score(info, sort_method="reward")
        macro = str_to_skills(data[0][0])
    
    env_id = serach_envID_from_path(path)
    rl_model = "a2c" if "a2c" in path else "ppo"
    assert env_id is not None
    config = [
        {
            "env_id":env_id,
            "rl_model":rl_model,
            "subdir":"../log/other/macro_ensemble/{}/dqn/{}/top".format(env_id, rl_model),
            "seed":2000,
            "macro":[
                macro,
            ]
        }
    ]
    print(config)
    return config


def get_cmd_from_exp_id(exp_id):
    # exp_id = exp_id.lower()
    if exp_id in all_exp:
        return all_exp[exp_id]
    elif os.path.isfile(exp_id) and "macro.txt" in exp_id:
        return generate_exp_config(exp_id)
    else: 
        raise NotImplementedError("{} not specify".format(exp_id))
  