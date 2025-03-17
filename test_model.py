"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
import json
import logging
from matplotlib import pyplot as plt
import torch
from src.evaluator import Evaluator
from models.RainbowNet import RainbowNet
from src.environment import AgentFighting
log = logging.getLogger(__name__)
from argparse import ArgumentParser
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

plt.ion()

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-s', '--show-screen', action='store_true', default=True)
    parser.add_argument('-n', '--n-evals', type=int, default=5)
    parser.add_argument('--model-path-1', type=str, default='trained_models/model.pt')
    parser.add_argument('--model-path-2', type=str, default='trained_models/model.v2.pt')
    parser.add_argument('--load-model', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.show_screen)
    observation_shape = env.get_space_size()
    n_actions = env.n_actions
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model_1 = RainbowNet(
        observation_shape, 
        n_actions, 
        v_min=configs['v_min'],
        v_max=configs['v_max'],
        atom_size=configs['atom_size'],
        device=device,
        version=1
    ).to(device)
    model_2 = RainbowNet(
        env.get_space_size(limit_obs_size=10), 
        n_actions, 
        v_min=configs['v_min'],
        v_max=configs['v_max'],
        atom_size=configs['atom_size'],
        device=device,
        version=2
    ).to(device)
    
    if args.load_model:
        model_1.load(args.model_path_1, device)
        model_2.load(args.model_path_2, device)
    
    evaluator = Evaluator(env, n_evals=args.n_evals, device=device)
    
    evaluator.eval(model_1, model_2, using_prob=False)

if __name__ == "__main__":
    main()