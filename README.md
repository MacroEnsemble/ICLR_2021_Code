
# Quick Run
## installation
```
bash shell_script/install.sh
```

## Construction Phase
```
cd DQN_skill

# python adqn.py [env-id] [version]
python3 adqn.py Asteroids 0
```

## Evaluation Phase
```
cd lib

# python3 atari_test.py [path/to/controller/log]
python3 atari_test.py ../log/controller/Asteroids/dqn/ppo/AsteroidsNoFrameskip_macro\[3\,3\]_r0.1_v0_dqn/macro/macro.txt
```

## Evaluate pretrain model
```
cd lib

# python3 evaluate_macros.py [path/to/pretrained/model]
python3 evaluate_macros.py ../log/macro_ensemble/Asteroids/dqn/ppo/top/\[\[4\,\ 3\,\ 1\]\,\ \[1\,\ 3\]\,\ \[1\,\ 0\,\ 0\]\]/model_1/model_1.pkl 
```
