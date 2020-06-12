# Toward Synergism in Macro Action Ensembles
This is a tensorflow implementation for paper "Toward Synergism in Macro Action Ensembles". The overview of the framwork is shown below.
![](https://i.imgur.com/WfeYZzK.png)



## Installation and Environment
The script installs all the prerequisite packages and  libraries . Note that if users build the environment with [virtualenv](https://virtualenv.pypa.io/en/latest/), please append the activating script(EX:`source activate /home/user/user_virtual_env/bin/activate`) to `shell_script/skill_search.sh` before running `install.sh`  

```bash
 # This script installs python 3.5 and other prerequisite library. 
 # The detail information for python libraries please refer to shell_script/requirements.txt
bash shell_script/install.sh
```


## Quick Run


### Macro Ensemble Construction 
```bash
cd DQN_skill/

# python adqn.py [env-id] [version]
python3 adqn.py Asteroids 0
```

### Evaluate an Constructed Macro Ensemble
```bash
cd lib/

# python3 atari_test.py [path/to/controller/log]
python3 atari_test.py ../log/controller/Asteroids/dqn/ppo/AsteroidsNoFrameskip_macro\[3\,3\]_r0.1_v0_dqn/macro/macro.txt
```

### Evaluate Pretrain Model
```bash
cd lib/

# python3 evaluate_macros.py [path/to/pretrained/model]
python3 evaluate_macros.py ../log/macro_ensemble/Asteroids/dqn/ppo/top/\[\[4\,\ 3\,\ 1\]\,\ \[1\,\ 3\]\,\ \[1\,\ 0\,\ 0\]\]/model_1/model_1.pkl 
```


## Experimental Results
We evaluate the constructed macro ensemble by training an RL agent based on the macro ensemble for 10M timesteps. The example results are demonstrated below.

![](https://i.imgur.com/NunC0aa.jpg)

