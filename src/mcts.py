import logging
import math
import numpy as np
EPS = 1e-8
log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game=None, model=None, numMCTSSims=15, selfplay=True, 
                 exploration_rate=0.25, cpuct=2.5):
        self.game = game
        self.model = model
        self.numMCTSSims = numMCTSSims
        self.selfplay = selfplay
        self.exploration_rate = exploration_rate
        self.cpuct_base = 19652
        self.cpuct_init = cpuct
        
        
        self.Qsa  = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa  = {}  # stores #times edge s,a was visited
        self.Ns   = {}  # stores #times board s was visited
        self.Ps   = {}  # stores initial policy (returned by neural net)

        self.Es   = {}  # stores game.get_game_ended ended for board s
        self.Vs   = {}  # stores game.getValidMoves for board s

    def reset(self):
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        
    def get_cpuct_value(self, s):
        cpuct = math.log((self.Ns[s] + self.cpuct_base + 1) / self.cpuct_base) + self.cpuct_init
        return cpuct

    def predict(self, board, temp=1):
        return self.getActionProb(board, temp)
        
    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        board = state.get_state()
        s = board['hash_str']
        for _ in range(self.numMCTSSims):
            self.search(state)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 
                  for a in range(self.game.n_actions)]
        # print([(a, self.Qsa[(s, a)]) for a in range(self.game.n_actions) if (s, a) in self.Qsa])
        # print([(a, self.Nsa[(s, a)]) for a in range(self.game.n_actions) if (s, a) in self.Nsa])
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
        else:
            # probs = softmax(1.0/temp * np.log(np.array(counts) + 1e-10))
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            if counts_sum == 0:
                probs = [1 / self.game.n_actions for _ in range(self.game.n_actions)]
            else:
                probs = [x / counts_sum for x in counts]
        if self.selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                valids = board['valid_actions']
                probs = np.array(probs) * valids
                if sum(valids) == 0:
                    probs = np.array([1 / self.game.n_actions] * self.game.n_actions)
                dirictlet_rd = valids * np.random.dirichlet(0.1 * np.ones(len(probs)))
                # renomalize dirictlet_rd to sum to 1
                dirictlet_rd = dirictlet_rd / np.sum(dirictlet_rd)
                # add dirictlet noise to probs
                probs = 0.9 * probs + 0.1 * dirictlet_rd
                probs = probs / np.sum(probs)
        return probs
    
    def get_action(self, state, temp=0):
        return np.argmax(self.getActionProb(state, temp))
    
    def predict(self, board):
        s = board.string_representation()
        for _ in range(self.numMCTSSims):
            self.search(board)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 
                  for a in range(self.game.n_actions)]
        # get probabilities of actions by counts
        sum_counts = float(sum(counts))
        probs = [x / sum_counts for x in counts]
        return probs
        

    def search(self, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current board
        """
        board = state.get_state()
        # self.game.render(state)
        s = board['hash_str']
        
        if state.terminal():
            scores = state.scores
            current_player = state.current_player
            R = 0
            if scores[current_player] > scores[1 - current_player]:
                R = 1 + np.log(1 + scores[current_player] - scores[1 - current_player])
            elif scores[current_player] < scores[1 - current_player]:
                R = - 1 - np.log(1 + scores[1 - current_player] - scores[current_player])
            if state.agent_current_idx == 0:
                return - R
            else:
                return R

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.model.get_pi_values(board['observation'])
            valids = board['valid_actions']
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            
            if np.sum(self.Ps[s]) == 0:
                if np.sum(valids) == 0:
                    self.Ps[s] = np.array([1 / self.game.n_actions] * self.game.n_actions)
                else:
                    self.Ps[s] = valids / np.sum(valids)  # if all valid moves were masked make all valid moves equally probable
                
            sum_Ps_s = np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            self.Ps[s] /= sum_Ps_s  # renormalize
            scores = state.scores
            current_player = state.current_player
            R = 0
            if scores[current_player] > scores[1 - current_player]:
                R = 1 + np.log(1 + scores[current_player] - scores[1 - current_player])
            elif scores[current_player] < scores[1 - current_player]:
                R = - 1 - np.log(1 + scores[1 - current_player] - scores[current_player])
            v = v * 0.6 + 0.4 * R
            if state.agent_current_idx == 0:
                return -v
            else:
                return v
     
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.n_actions):
            if valids[a]:
                cpuct = self.get_cpuct_value(s)
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                # print(a, u)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        
        next_state = state.copy()
        next_state.next(a)
        v = self.search(next_state)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) \
                / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        if state.agent_current_idx == 0:
            return - v
        else:
            return v