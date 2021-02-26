import math
import numpy as np
import time
import random
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

        self.Fp = {}
        self.Fv = {}
        self.rate = {}
        self.timetotal = 0
        self.book_dict = np.load('./openbook.npy',allow_pickle='True').item()


    def getActionProb(self, canonicalBoard, temp=1):
        valids = self.game.getValidMoves(canonicalBoard, 1)
        if valids[self.game.getActionSize()-1] == 1:
            probs = [0]*self.game.getActionSize()
            probs[self.game.getActionSize()-1]=1
            return probs

        countstep = 0
        for i in range(self.game.n):
            for j in range(self.game.n):
                if(canonicalBoard[i][j] == 0):
                    countstep += 1
        

        if(countstep <= 12):
            result = self.ABS(canonicalBoard)
            return result
        
        else:
            s = self.game.stringRepresentation(canonicalBoard)
            
            
                
            if s in self.book_dict:
                print('use dict')
                a = random.choice(self.book_dict[s])
                
                probs = [0]*self.game.getActionSize()
                probs[a]=1
                return probs
            
            for i in range(self.args.numMCTSSims):
                self.search(canonicalBoard)
            

            

            counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
            if temp==0:
                bestA = np.argmax(counts)
                probs = [0]*len(counts)
                probs[bestA]=1

                return probs

            counts = [x**(1./temp) for x in counts]

            probs = [x/float(sum(counts)) for x in counts]
            
            return probs

    def maxv(self, canonicalBoard, bot, alpha, beta):
       
        result = self.game.getGameEnded(canonicalBoard, 1)
        if result != 0:
            
            if(result == -2):
                return 0
            if(bot == 1):
                return result
            else:
                return -result
        
        valids = self.game.getValidMoves(canonicalBoard, 1)
        score = -float('inf')
        best_action = -1
        for a in range(self.game.getActionSize()):     
            if valids[a]:
                next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
                next_s = self.game.getCanonicalForm(next_s, next_player)
                
                #score = max(score, self.minv(next_s, -bot, alpha, beta))
                s = self.minv(next_s, -bot, alpha, beta)
                if s > score:
                    score = s
                    best_action = a
                
                if score >= beta:
                    
                    return score
                alpha = max(alpha, score)
                if score == 1:
                    return score

        return score

    def minv(self, canonicalBoard, bot, alpha, beta):
        
        result = self.game.getGameEnded(canonicalBoard, 1)
        if result != 0:
            if(result == -2):
                return 0
            if(bot == 1):
                return result
            else:
                return -result
        valids = self.game.getValidMoves(canonicalBoard, 1)
        score = float('inf')
        for a in range(self.game.getActionSize()):     
            if valids[a]:
                next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
                next_s = self.game.getCanonicalForm(next_s, next_player)
                score = min(score, self.maxv(next_s, -bot, alpha, beta))
                if score <= alpha:
                    return score
                beta = min(beta, score)
                
                if(score == -1):
                    return score
        
        return score
    
    def ABS(self, canonicalBoard):
        
        #print('new situation')
        valids = self.game.getValidMoves(canonicalBoard, 1)
        alpha = -float('inf')
        beta = float('inf')
        score = -float('inf')
        best_action = -1
        
        
        for a in range(self.game.getActionSize()):     
            if valids[a]:
                next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
                next_s = self.game.getCanonicalForm(next_s, next_player)
                s = self.minv(next_s, next_player, alpha, beta)
                if s > score:
                    score = s
                    best_action = a
                    
                alpha = max(alpha, score)
        '''
        if score == 1:
            print('win')
        else:
            print('thinking...')
        '''
        probs = [0]*self.game.getActionSize()
        probs[best_action] = 1
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        
        s = self.game.stringRepresentation(canonicalBoard)
        
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s]!=0:
            # terminal node
            if (self.Es[s] == -2):
                return 0
            return -self.Es[s]

        if s not in self.Ps:
            discountstep = 0
            for i in range(self.game.n):
                for j in range(self.game.n):
                    if(canonicalBoard[i][j] == 0):
                        discountstep += 1
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            
            v1, step = self.future_v(canonicalBoard)
            if step != 0:
                v = v1 / step
            
            valids = self.game.getValidMoves(canonicalBoard, 1)
            
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    if (s,a) in self.rate:
                        u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)] + self.rate[(s,a)])
                    else:
                        u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        

        v = self.search(next_s)

        if(v == -1 or v == 0):
            
            if (s,a) in self.rate:
                self.rate[(s,a)] += 8
            else:
                self.rate[(s,a)] = 8

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v


    def future_v(self, canonicalBoard, countstep = 0, simulation_step = 0):
        if simulation_step == 2:
            return 0, simulation_step
        if(countstep == 0):
            for i in range(self.game.n):
                for j in range(self.game.n):
                    if(canonicalBoard[i][j] == 0):
                        countstep += 1
        valids = self.game.getValidMoves(canonicalBoard, 1)

        s = self.game.stringRepresentation(canonicalBoard)
        if(countstep > 12):
            
            if s not in self.Fp:
                self.Fp[s], self.Fv[s] = self.nnet.predict(canonicalBoard)

                self.Fp[s] = self.Fp[s]*valids
                sum_p = np.sum(self.Fp[s])
                if sum_p > 0:
                    self.Fp[s] /= sum_p 
            
            a = np.argmax(self.Fp[s])
            
            next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
            next_s = self.game.getCanonicalForm(next_s, next_player)
            if(a != self.game.getActionSize()):
                countstep -= 1
            simulation_step += 1
            back_v, simulation_r = self.future_v(next_s, countstep, simulation_step)
            if simulation_step % 2 == 0:
                back_v -= self.Fv[s]
            else:
                back_v += self.Fv[s]
            return back_v, simulation_r
        else:
            return 0, simulation_step