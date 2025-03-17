"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
import json
import logging
import time

import torch
from algorithms.RandomStep import RandomStep
from algorithms.Minimax import Minimax
from models.RainbowNet import RainbowNet
from src.environment import AgentFighting
log = logging.getLogger(__name__)
from random import seed
from argparse import ArgumentParser

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-s', '--show-screen', action='store_true')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--model-path', type=str, default='trained_models/nnet3.pt')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.show_screen)
    observation_shape = env.get_space_size()
    n_actions = env.n_actions
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model = RainbowNet(
        observation_shape, 
        n_actions, 
        v_min=configs['v_min'],
        v_max=configs['v_max'],
        atom_size=configs['atom_size'],
        device=device
    ).to(device)
    
    # model.load(args.model_path)
    
    algorithm = Minimax(env, model, max_depth=2, handicap=2)
    
    if args.show_screen:
        env.render()
    env.save_image('figures/minimax.png')
    state = env.get_state(obj=True)
    while not env.is_terminal():
        action = algorithm.get_action(state)
        _ = env.step(action, verbose=True)
        env.save_image('figures/minimax.png')
        state = env.get_state(obj=True)
    
    winner = env.get_winner()
    if winner == -1:
        logging.info('Game ended. Draw')
    else:
        logging.info('Game ended. Winner: {}'.format(env.get_winner()))

if __name__ == "__main__":
    main()