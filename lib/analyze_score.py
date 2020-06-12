import argparse
import os
import operator


def get_info(log_path):
    info=dict()
    with open(log_path, 'r') as f:
        content = f.read().split("\n")
        # print(content)
        for idx, ele in enumerate(content):
            if ":" in ele:
                key_value = ele.split(":")
                key = key_value[0]
                try: 
                    value = float(key_value[1])
                except ValueError:
                    value = key_value[1]

                info.update({key:{"reward":value, "idx":idx}})
    return info



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


"""
skills:[[5, 2, 2], [-1, 4, 4], [4, 3, 4]], reward:10.5, epsilon:0.22615769712140166
skills:[[5, 2, 2], [-1, 4, 4], [4, 2, -1]], reward:10.8, epsilon:0.22615769712140166
...
"""
import re
def get_maro_info(log_path):
    info = dict()
    with open(log_path, 'r') as f:
        content = f.read().split("\n")
        # content=content.replace(" ","").split("\n")
    for idx, ele in enumerate(content):
        # key_values = ele.split(",")
        skills = None
        reward = None
        # delimenator="[a-zA-Z]+:[0-9.,\[(\])]+"
        search="[\w]+:[0-9\[\](0-9|\-, ).]+"
        # print(ele)
        # exit(0)
        find_list=re.findall(search, ele)
        # print(find_list)
        for kv in find_list:
            # print(kv)
            
            if "skills" in kv:
                if kv[-2:]==", ":
                    kv=kv[:-2]
                skills = kv.replace("skills:", "")
                # print(skills)
            elif "reward" in kv:
                if kv[-2:]==", ":
                    kv=kv[:-2]
                reward = kv.replace("reward:", "")
                reward = toNumber(reward)
            if skills is not None and reward is not None:
                if skills in info:
                    # print("duplicate")
                    pass  
                # info.update({skills:reward})
                info.update({skills:{"reward":reward, "idx":idx}})

    return info



def sort_score(dict_, num=1, sort_method="reward"):
    assert num>0
    # sorted_dict = dict_
    
    # sorted_dict = sorted(dict_.items(), key=operator.itemgetter(1), reverse=True)
    if sort_method == "reward":
        sorted_dict = sorted(dict_.items(), key=lambda x: x[1]["reward"], reverse=True)
    elif sort_method =="index":
        sorted_dict = sorted(dict_.items(), key=lambda x: x[1]["idx"], reverse=True)
    
    # print(sorted_dict[:num])
    if num>len(sorted_dict):
        return sorted_dict
    else:
        return sorted_dict[:num]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_file", type=str)
    parser.add_argument("--top", default=10, type=int)
    args = parser.parse_args()
    assert os.path.exists(args.score_file)
    data_type = None
    if "macro" in args.score_file:
        d=get_maro_info(args.score_file)
    elif "score" in args.score_file:
        d=get_info(args.score_file)
    else:
        raise ValueError("{} file should be either macro.txt or score.txt".format(args.score_file))

    top=sort_score(d,args.top)
    for item in top:
        print(item)
    print("{}/{}".format(len(top), len(d)))
if __name__ == "__main__":
    main()