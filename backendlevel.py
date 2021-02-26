from flask import Flask, request
from flask_cors import CORS
import json
from lastMCTST import MCTS as MCTS
from MCTS import MCTS as MCT
import numpy as np
from othello.OthelloGame import OthelloGame
from othello.keras.newNNet import NNetWrapper as NNet
from utils import *
import tensorflow as tf



app=Flask(__name__)
CORS(app, resources=r'/*')



game = OthelloGame(8)
hnnet = NNet(game)
ennet = NNet(game)
hnnet.load_checkpoint('./temp/','hardbest.pth.tar')
ennet.load_checkpoint('./temp/','easybest.pth.tar')
hargs = dotdict({'numMCTSSims': 500, 'cpuct':1.0})
eargs = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
hmcts = MCTS(game, hnnet, hargs)
emcts = MCT(game, ennet, eargs)


@app.route('/test', methods=['GET', 'POST'])
def test():
    dic = [[19,26,37,44],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0],[0]]
    pointX = request.form['pointX']
    pointY = request.form['pointY']
    data = request.form.get("data")
    color = request.form['color']
    rank = request.form['rank']
    D_list = json.loads(data)


    if int(rank) == 0:
        mcts = emcts
    elif int(rank) == 1:
        mcts = hmcts
    
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