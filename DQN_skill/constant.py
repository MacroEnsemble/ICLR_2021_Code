general_constant = {
    # Arguments of experiment
    "NUM_SKILL" : 3,
    "MAXLEN_PER_SKILL" : 3,
    # LEN_SKILL : NUM_SKILL * MAXLEN_PER_SKILL # Total Length of skills e.g (2,3):>6, (1,9):>9
    "TOTAL_NUM_OF_WORKER" : 30,
    # UPDATE_FREQ : LEN_SKILL * TOTAL_NUM_OF_WORKER # Total Length of skills training at the same time e.g 20 woker with macro shape(2,3):> 20*2*3:120

    # Hyperparameters
    "NUM_EPOCH" : 2,
    "LEARNING_RATE" : 5e-6,
    "PENALTY" : -1000,
    "BATCH_SIZE" : 128,

    "NUM_INIT_SKILL" : 200,
    "NUM_EXPLORE_SKILL": 800,
    "MAX_NUM_TOTAL_SKILL" : 1201,
    "MAX_LIMIT_TIME": 4*86400, 
    # Worker parameters
    "WORKER_TRAINING_STEP" : int(5e6),
}


env_list = [    "Alien", "Seaquest", "BeamRider",
                "Breakout", "SpaceInvaders", "KungFuMaster",
                "Venture",  "Asteroids", "Gravitar",
                "Solaris", "Frostbite", "CrazyClimber", "Zaxxon"]

# s

Alien_constant = {
    "REWARD_FACTOR" : 0.05,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5],
}

Seaquest_constant = {
    "REWARD_FACTOR" : 0.1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5],
}

BeamRider_constant = {
    "REWARD_FACTOR" : 0.05,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3],
}

Breakout_constant = {   
    "REWARD_FACTOR" : 0.05,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3],
}



SpaceInvaders_constant = {
    "REWARD_FACTOR" : 0.05,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3],
}

KungFuMaster_constant = {
    "REWARD_FACTOR" : 0.02,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
}


Venture_constant = {
    "REWARD_FACTOR" : 1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4],
    "WORKER_TRAINING_STEP": int(5e6),
}


Asteroids_constant = {
    "REWARD_FACTOR" : 0.1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5, 6, 7],
}

Gravitar_constant = {
    "REWARD_FACTOR" : 1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}

Frostbite_constant = {
    "REWARD_FACTOR" : 1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}
Solaris_constant = {
    "REWARD_FACTOR" : 1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}

CrazyClimber_constant = {
    "REWARD_FACTOR" : 0.1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
} 


Zaxxon_constant = {
    "REWARD_FACTOR" : 0.1,
    "VALID_ACTIONS" : [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
}




Alien_constant.update(general_constant)

Asteroids_constant.update(general_constant)
BeamRider_constant.update(general_constant)
Breakout_constant.update(general_constant)
CrazyClimber_constant.update(general_constant)
Frostbite_constant.update(general_constant)
Gravitar_constant.update(general_constant)
KungFuMaster_constant.update(general_constant)
Seaquest_constant.update(general_constant)
SpaceInvaders_constant.update(general_constant)
Solaris_constant.update(general_constant)
Venture_constant.update(general_constant)
Zaxxon_constant.update(general_constant)

if __name__ == "__main__":
    print(Seaquest_constant)
    
    
