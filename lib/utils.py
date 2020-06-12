

import numpy as np
def random_init_skills(num_skills, len_range, candidate_action):
    """"
    :param num_skills: (int) how many skills to generate
    :param len_range: (tuple) shold be the form of [MIN, MAX],  MIN should larger than 1
    :param candidate_action: (list) candidate action for action sequence
    return: (list)skills
    """
    assert len(len_range) == 2
    assert len_range[0] >=1

    if num_skills<=len(candidate_action):
        rand_action_base = np.random.choice(candidate_action, num_skills, replace=False).tolist()
    else:
        rand_action_base = np.random.choice(candidate_action, num_skills, replace=True).tolist()

    rand_length = [np.random.randint(len_range[0], len_range[1]+1, num_skills)]

    new_skills = [ [ rand_action_base[i] for j in range(rand_length[i]) ] for i in range(num_skills)]
    return new_skills

# [4, 3, 3, 3, 0, 4, 
#  0, 0, 2, 2, 5, 2, 
#  4, 2, 0, 0, 3, 2, 
#  4, 3, 0, 5, 0, 3]

def _state_to_skills(state, max_skill_len, noop=0):
    skills = []
    skill = []
    for i, act in enumerate(state):
        if act!=noop:
            skill.append(act)
        
        if (i+1) % max_skill_len == 0:
            if len(skill) != 0:
                skills.append(skill)
            skill=[]
    return skills
    
def check_duplicate(skills):
    for i in range(len(skills)):
        for j in range(i+1, len(skills)):
            if(skills[i]==skills[j]):
                return True
    return False


def toNumber(num, return_none=True):
    result=None
    try:
        result=int(num)
    except ValueError:
        try:
            result=float(num)
        except ValueError:
            if return_none is not True:
                raise
    return result


import errno
import os, shutil
import time
def mkdirs(path, mode="ignore"):
    """
    :param path: (str) directory path
    :param keep: (str) ignore/keep/interactive/remove
        - ignore: ignore if dir already exists
        - keep: append serial number to the dir if dir already exists
        - interactive: let the user decide interactivly 
        - remove: remove the file if exists
    :return path of maked directory
    """
    idx = 0
    if path[-1] == "/":
        path = path[:-1]
    # print(path)
    final_path=path
    while True:
        try:
            os.makedirs(final_path)
            break
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                if mode=="keep":
                    final_path = "{}_{}".format(path, idx)
                    idx += 1
                elif mode=="ignore":
                    break
                elif mode=="remove":
                    shutil.rmtree(path)
                elif mode=="interactive":
                    while True:
                        ans=input("Are you sure to remove {} (rm/keep/ignore)? ".format(path))
                        if ans.lower()=='rm' or ans.lower()=='r':
                            shutil.rmtree(path, ignore_errors=True)
                            break
                        elif ans.lower()=='keep'or ans.lower()=='k':
                            mode="keep"
                            break
                        elif ans.lower()=='ignore' or ans.lower()=='i':
                            # raise OSError("{} exists".format(path))
                            # print("{} exists".format(path))
                            # exit(0)
                            mode="ignore"
                            break
                else:
                    raise ValueError("{} is not the valid mode(ignore/keep/remove/interactive)".format(mode))
            elif exc.errno==errno.EBUSY:
                print("resource device busy, wait 5 sec and retry...")
                time.sleep(5)
            else:
                raise
    return final_path

import GPUtil
def set_GPU(delay_time=30):
    while True:
        deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
        if len(deviceIDs)==0:
            print("No GPU available, wait for {} sec".format(delay_time))
            time.sleep(delay_time)
        else:
            print("export CUDA_VISIBLE_DEVICES={}".format(deviceIDs[0]))
            os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(deviceIDs[0])
            break

    
def str_to_skills(str_skills):
    str_skills = str_skills.replace(" ", '')
    
    skills = []
    temp_idx = 0
    # print(str_skills[1:-1])
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


def test_mkdirs():
    mkdirs("./test_dir/")
    final_path = mkdirs("./test_dir", mode="ignore")
    if os.path.exists(final_path):
        print("[PASS] ignore")
        shutil.rmtree(final_path)
    else:
        print("[FAIL] ignore")
    

    mkdirs("./test_dir")
    final_path=mkdirs("./test_dir", mode="keep")
    if os.path.exists(final_path):
        print("[PASS] keep")
        shutil.rmtree(final_path)
    else:
        print("[FAIL] keep")
    

    mkdirs("./test_dir")
    final_path = mkdirs("./test_dir", mode="remove")
    if os.path.exists(final_path):
        print("[PASS] remove")
        shutil.rmtree(final_path)
    else:
        print("[FAIL] remove")
    

    mkdirs("./test_dir")
    final_path = mkdirs("./test_dir", mode="interactive")
    if os.path.exists(final_path):
        print("[PASS] interacive")
        shutil.rmtree(final_path)
    else:
        print("[FAIL] interacive")
def test_check_dupilcate():
    assert check_duplicate([[1,2,2],[4,2,3,1],[1,2,2]]) is True
    assert check_duplicate([[0,0,0],[0,0,0]]) is True
    assert check_duplicate([[0,0,0]]) is False
    assert check_duplicate([[0,0,0],[0,0]]) is False

def path_formalize(path):
    new_path = path.replace('[', '&#91;').replace(']', '&#93;').replace('&#91;', '[[]').replace('&#93;', '[]]')
    return new_path

def test_path_formalize():
    pass



def main():
    # test_check_dupilcate()
    test_mkdirs()
    pass
if __name__ == "__main__":
    main()
