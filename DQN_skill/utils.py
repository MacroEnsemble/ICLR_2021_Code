import os
import numpy as np

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

def read_score(log_path, remove=True):
    """
    Args:
        log_path:
            (str)
    
    Returns:
        [ensemble, reward, epsilon]:
            
    """
    result = [None, None, None]
    for root, dirs, files in os.walk(log_path):     # everything in the tree
        for file in files:
            if file.endswith(".txt"):
                score_path=os.path.join(root, file)
                
                with open(score_path) as f:
                    content = f.read().split("\n")
                    if len(content)<1:      # Null file
                        continue
                    for line in content:
                        if ":" in line and "delta" not in line:
                            key, value = line.split(":")
                            if key == "skill":
                                result[0] = str_to_skills(value)
                            elif key == "score":
                                result[1] = float(value)
                            elif key == "info":
                                result[2] = float(value)
                        else:
                            continue

                    
                if remove is True:
                    os.remove(score_path)
                    
                return result
    return result

def get_latest_ckpt(path):
    '''
    Args:
        path:
            The list containing path of checkpoints. e.g.['model_0.ckpt', 'model_100.ckpt', ... ]
    '''
    assert type(path) == list
    
    ckpt_idx = -1
    ckpt = None
    
    for p in path:
        if '.ckpt' not in p:
            continue
        idx = int(p[6:p.find('.ckpt')])
        if idx > ckpt_idx:
            ckpt_idx = idx
            # ckpt = p
    
    if ckpt_idx != -1:
        ckpt = 'model_{}.ckpt'.format(ckpt_idx)

    return ckpt_idx, ckpt

if __name__ == '__main__':
    path = ['model_0.ckpt', 'model_100.ckpt', 'model_101.ckpt']
    print(get_latest_ckpt(path))

        
        
    