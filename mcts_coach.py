"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
from __future__ import division
import argparse

import json
import logging
import os

import torch
from Algorithms.MCTS import MonteCarloTrainer
from models.AgentDQN import DQN
from models.AgentPG import AlphaZeroNet
from src.environment import AgentFighting
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='', 
                        help='name of the model')
    parser.add_argument('--show_screen', action='store_true', default=True,
                        help='show the screen')
    parser.add_argument('--exploration_rate', type=float, default=0.1, 
                        help='exploration rate for self-play')
    parser.add_argument('--exp_rate', type=float, default=0.2, 
                        help='experimental rate')
    parser.add_argument('--_is_selfplay', type=bool, default=True,
                        help='if true, then self-play, else, then test')
    parser.add_argument('--numIters', type=int, default=1000,
                        help='number of iterations')
    parser.add_argument('--nCompare', type=int, default=50, 
                        help='Number of games to play during arena play to determine if new net will be accepted.')
    parser.add_argument('--numEps', type=int, default=5,
                        help='Number of complete self-play games to simulate during a new iteration.')
    parser.add_argument('--tempThreshold', type=int, default=10, 
                        help='tempThreshold')
    parser.add_argument('--updateThreshold', type=float, default=0.5,
                        help='During arena playoff, new neural net will be accepted if threshold or more of games are won.')
    parser.add_argument('--maxlenOfQueue', type=int, default=500000,
                        help='Number of game examples to train the neural networks.')
    parser.add_argument('--numMCTSSims', type=int, default=40, 
                        help='Number of games moves for MCTS to simulate.')
    parser.add_argument('--cpuct', type=float, default=1.25, 
                        help='a heuristic value used to balance exploration and exploitation.')
    parser.add_argument('--checkpoint', type=str, default='./temp/', 
                        help='Directory to save the checkpoints.')
    parser.add_argument('--trainEpochs', type=int, default=5,
                        help='Number of epochs to train the neural network.')
    parser.add_argument('--trainBatchSize', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--loss_func', type=str, default='mse',
                        help='Loss function for training.')
    parser.add_argument('--load_folder_file', type=list, default=['trained_models','alphanet.pt'], 
                        help='(folder,file) to load the pre-trained model from.')
    parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=50,
                        help='Number of iterations to store the trainExamples history.')
    parser.add_argument('--saved_model', action='store_true', default=False,  
                        help='Whether to save the model.')
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=32768 * 4)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--load-model', action='store_true', default=False)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    configs = json.load(open('configs/map.json'))
    env = AgentFighting(args, configs, args.show_screen)
    observation_shape = env.get_space_size()
    n_actions = env.n_actions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    players = [
        AlphaZeroNet(observation_shape, n_actions, 
                optimizer=args.optimizer, 
                lr=args.lr).to(device),
        AlphaZeroNet(observation_shape, n_actions, 
                optimizer=args.optimizer, 
                lr=args.lr).to(device),
        # DQN(observation_shape, n_actions, 
        #         optimizer=args.optimizer, 
        #         lr=args.lr, dueling=True)
    ]
    
    if args.load_model:
        players[0].load(os.path.join(args.load_folder_file[0], args.load_folder_file[1]), device)
        players[1].load(os.path.join(args.load_folder_file[0], args.load_folder_file[1]), device)
        # players[2].to(device).load('trained_models/nnet2.pt', device)
            
    coach = MonteCarloTrainer (
        game=env, 
        players=players,
        numEps=args.numEps, 
        tempThreshold=args.tempThreshold,
        updateThreshold=args.updateThreshold,
        maxlenOfQueue=args.maxlenOfQueue,
        numMCTSSims=args.numMCTSSims,
        exploration_rate=args.exploration_rate,
        cpuct=args.cpuct,
        show_screen=args.show_screen,
        numItersForTrainExamplesHistory=args.numItersForTrainExamplesHistory,
        checkpoint=args.checkpoint,
        train_epochs=args.trainEpochs,
        batch_size=args.trainBatchSize,
        loss_func=args.loss_func,
        n_compares=args.nCompare,
        load_folder_file=args.load_folder_file,
    )
        
    # if args.load_model:````
    # coach.loadTrainExamples()

    for i in range(0, args.numIters):
        # bookkeeping
        print(f'Starting Iter #{i} ...')
        coach.learn(i)
        
if __name__ == "__main__":
    main()