import sys
import time
import math
import random
import numpy as np
import gym

from gym import spaces, logger

class DeitEnv(gym.Env):
    """
    Description:

    Source:
        This environment corresponds to Deit
        described by ludashuai

    Observation: 
        Type: Discrete(6)
        Num	Observation                
        0   History Temperature	minus Mean 
        1	History Temperature	minus Mean 
        2	History Temperature	minus Mean   
        3	History Temperature	minus Mean                      
        4	History Temperature	minus Mean                             
        5   Mean of History Temperature
        
    Actions:
        Type: Discrete(2)
        Num	Action          Min         Max
        0   Fan Speed       2000	    10000
        

    Reward:
        Reward is current temperature minus target temperature for every step taken,
        including the termination step

    Starting State:

    Episode Termination:
        Considered solved when the total step bigger than 1000.
    """
    def __init__(self, condition):
        self.name = "#####"
        self.condition = condition
        self.obs_ = None
        pass

    def step(self, action):
        # wait for singnal
        # deit run step_one_block
        # deit send continue singnal
        # deit send [obs, action, obs_]
        # receive [obs, action, obs_] from deit
        # continue
        # with self.condition:
        #     print("{} : Mr.Deit, r u finish step one block?".format(self.name)
        #     condition.notify()
        #     condition.wait()

        #     print("{} : Mr.Deit, r u finish set obs next?".format(self.name)
        #     condition.notify()
        #     condition.wait()
        #     
        #     print("{} : Trying to get obs next ... ".format(self.name))
        #     obs_ = self.get_obs_next()
        #     print("{} : Got it!".format(self.name))

        return obs_, reward, done, {}


    def set_obs_next(self):
        pass
    
    def get_obs_next(self):
        pass

    def reset(self):
        pass

    def close():
        pass

