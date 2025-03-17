"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from __future__ import division
from collections import deque
from copy import deepcopy
from itertools import count
import json
import logging
import os
import time
import torch
from tqdm import tqdm
from src.evaluator import Evaluator
from src.environment import AgentFighting
from src.utils import *
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
from argparse import ArgumentParser
from algorithms.Rainbow import Rainbow
from models.RainbowNet import RainbowNet

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-s', '--show-screen', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--figure-path', type=str, default='figures/')
    parser.add_argument('--n-evals', type=int, default=50)
    
    # DDQN arguments
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--n-step', type=int, default=3)
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=32768 * 2)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--model-path', type=str, default='trained_models/model.pt')
    parser.add_argument('--load-model', action='store_true')
    
    return parser.parse_args()

def main():
    args = argument_parser()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.show_screen)
    observation_shape = env.get_space_size(limit_obs_size=7)
    n_actions = env.n_actions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RainbowNet(
        observation_shape, 
        n_actions, 
        optimizer=args.optimizer, 
        lr=args.lr,
        atom_size=configs['atom_size'], 
        v_min=configs['v_min'],
        v_max=configs['v_max'],
    ).to(device)
    
    if args.load_model:
        model.load(args.model_path, device)
        
    evaluator = Evaluator(AgentFighting(args, configs, False), n_evals=args.n_evals, device=device)
    
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info('Created model directory: {}'.format(model_dir))

    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        logging.info('Created figure directory: {}'.format(args.figure_path))
        
    algorithm = Rainbow(observation_shape=observation_shape, 
                        n_actions=n_actions,
                        model=model,
                        tau=args.tau,
                        gamma=args.gamma,
                        memory_size=args.memory_size,
                        model_path=args.model_path,
                        batch_size=args.batch_size,
                        alpha=0.2,
                        beta=0.6,
                        prior_eps=1e-6,
                        n_step=args.n_step,
                        atom_size=configs['atom_size'], 
                        v_min=configs['v_min'],
                        v_max=configs['v_max'],
                    )
    
    algorithm.set_multi_agent_env(env.num_agents)
    
    best_model_path = args.model_path.replace('.pt', '_best.pt')
    model.save(best_model_path)
    best_model = deepcopy(model)
    best_model.load(best_model_path, device)
    target_player_id = 1
    
    for episode in range(args.num_episodes):
        _tqdm = tqdm(range(10), 'Self-Play')
        for i_game in _tqdm:
            done = False
            state = env.get_state()
            observations = []
            local_rewards = []
            actions = []
            prev_diff_scores = []
            
            for cnt in count():
                env.render()
                obs = state['observation']
                valid_actions = state['valid_actions']
                prev_diff_score = env.get_diff_score()
                if target_player_id == state['player-id']:
                    action = algorithm.get_action(obs, valid_actions, model=best_model)
                    next_state, reward, done = env.step(action)
                    next_obs = next_state['observation']
                else:
                    action = algorithm.get_action(obs, valid_actions)
                    next_state, reward, done = env.step(action)
                    next_obs = next_state['observation']
                    observations.append(obs)
                    local_rewards.append(reward)
                    actions.append(action)
                    prev_diff_scores.append(prev_diff_score)
                    
                next_agent_idx = env.get_curr_agent_idx()
                if target_player_id == state['player-id'] and next_agent_idx == 0:
                    curr_diff_score = env.get_diff_score()
                    # env.save_image('figures/live_update.png')
                    for obs, action, reward, prev_diff_score in zip(observations, actions, local_rewards, prev_diff_scores):
                        global_reward = (curr_diff_score - prev_diff_score) / env.num_agents
                        # obs, action, next_obs = env.get_symmetry_transition(obs, action, next_obs)
                        reward = reward * 0.975 + global_reward * 0.025
                        transition = [obs, action, reward, next_obs, False]
                        one_step_transition = algorithm.memory_n.store(*transition)
                        if one_step_transition:
                            algorithm.memory.store(*one_step_transition)
                        algorithm.memorize(*transition)
                        
                    observations = []
                    local_rewards = []
                    actions = []
                    prev_diff_scores = []
                    prev_diff_score = curr_diff_score
                    
                state = next_state
                if done:
                    break
                
            env.reset()
            algorithm.memory.free_buffer()
            
        history_loss = algorithm.replay(args.batch_size, verbose=args.verbose)
        plot_timeseries(history_loss, args.figure_path, 'episode', 'loss', 'Training Loss')
        
        if (episode + 1) % 10 == 0:
            model.save(args.model_path)
            best_model.load(best_model_path, device)
            improved = evaluator.eval(best_model, model)
            if improved:
                model.save(best_model_path)
                best_model.load(best_model_path, device)
                best_model.save(best_model_path.replace('.pt', '_base.pt'), metadata=False)
                
            plot_timeseries(model.elo_history, args.figure_path, 'episode', 'elo', 'Elo')
            

if __name__ == "__main__":
    main()