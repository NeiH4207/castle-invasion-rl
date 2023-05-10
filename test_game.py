"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
import json
import logging
import time
from Algorithms.RandomStep import RandomStep
from src.environment import AgentFighting
log = logging.getLogger(__name__)
from random import seed
from argparse import ArgumentParser

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--render', type=bool, default=True)
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('config.json'))
    env = AgentFighting(args, configs)
    
    env.reset()
    algorithm = RandomStep(num_actions=env.num_actions, num_agents=env.num_agents)
    while not env.game_ended():
        state = env.get_state()
        action = algorithm.get_action(state)
        reward = env.step(action)
        env.render()
        time.sleep(0.05)
        

if __name__ == "__main__":
    main()