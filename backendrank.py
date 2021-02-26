from flask import Flask, request
from flask_cors import CORS
import json
from lastMCTST import MCTS as MCTS
import numpy as np
from othello.OthelloGame import OthelloGame
from othello.keras.newNNet import NNetWrapper as NNet
from utils import *
import tensorflow as tf

game = OthelloGame(8)

app=Flask(__name__)
CORS(app, resources=r'/*')

change_rank = -1
mcts = None

@app.route('/test', methods=['GET', 'POST'])
def test():
    dic = [[19,26,37,44],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0],[0]]
    pointX = request.form['pointX']
    pointY = request.form['pointY']
    data = request.form.get("data")
    color = request.form['color']
    rank = request.form['rank']
    D_list = json.loads(data)
    global change_rank
    global mcts
    if change_rank != int(rank):
        if int(rank) == 0:
            
            nnet = NNet(game)
            nnet.load_checkpoint('./temp/','easybest.pth.tar')
            args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
            mcts = MCTS(game, nnet, args)
            change_rank = int(rank)
        elif int(rank) == 1:
            
            nnet = NNet(game)
            nnet.load_checkpoint('./temp/','hardbest.pth.tar')
            args = dotdict({'numMCTSSims': 400, 'cpuct':1.0})
            mcts = MCTS(game, nnet, args)
            change_rank = int(rank)
        else:
            
            nnet = NNet(game)
            nnet.load_checkpoint('./temp/','easybest.pth.tar')
            args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
            mcts = MCTS(game, nnet, args)
            change_rank = int(rank)

    dic[1] = D_list
    dic[0] = []
    dic[2] = 0
    dic[3] = [0]
    if int(color) != 0:
        board = np.reshape(D_list, (8,8))
        curPlayer = int(color)
        
        if int(pointX) == -1 and int(pointY) == -1:
            agent = curPlayer * -1
            canonicalBoard = game.getCanonicalForm(board, agent)  #轉換盤面
            pi = mcts.getActionProb(canonicalBoard, temp = 0)  #MCTS
            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, agent, action)#下一狀態
            valids = game.getValidMoves(board, curPlayer)
            dic[3] = action
            for i in range(len(valids)):
                if valids[i]:
                    dic[0].append(i)
        else:
            action = (int(pointY))*8+(int(pointX))
            board, curPlayer = game.getNextState(board, int(color), action)  #下一狀態
            dic[3] = action


        r = game.getGameEnded(board, 1)
        if r!=0:
            dic[2] = r
        for i in range(8):
            for j in range(8):
                dic[1][i*8+j] = int(board[i][j])

        return json.dumps(dic)


if __name__=='__main__':
    app.run('0.0.0.0', port=80)