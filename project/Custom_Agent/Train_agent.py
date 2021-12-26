from collections import deque
import random

import chess
import numpy as np

from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent, QLearning
from project.Custom_Agent.Neural_net import create_utilitymodel

epochs = 10000
batchsize = 32
discount = 0.9
decay = 0.999

winreward = 1000000
drawreward = 100
stepreward = -1

utility = BoardUtility()

agent = ChessAgent(utility, 1)
epsilon = 0.8

policyModel = create_utilitymodel()
targetModel = create_utilitymodel()
deepq = QLearning(policyModel, targetModel, batchsize, discount, epsilon, decay)

training_dataset = []

targetupdatefreq = 5000
targetupdatecounter = 0
reward_count = [0, 0, 0]

for i in range(epochs):
    board = chess.Board()
    boardValue = BoardUtility.one_hot_board(board)
    pgn = ""
    for _ in range(300):
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            break
        # whites move
        move, newBoardValue, color = deepq.GetAction(board)
        board.push(move)
        pgn += "%s. %s " % (i, move.uci())

        if board.is_checkmate():
            reward = winreward
            done = True
            reward_count[0] += 1
        elif board.is_stalemate() or board.is_insufficient_material():  # board.is_seventyfive_moves() #board.is_fivefold_repetition() nog twee exit condidtions,
            reward = drawreward
            done = True
            reward_count[1] += 1
        else:
            reward = stepreward
            done = False
            reward_count[2] += 1

        deepq.AddToMemory(boardValue, reward, newBoardValue, done, color)
        boardValue = newBoardValue

        deepq.UpdatePolicyModel()
        print("\rTraining episode: %s, prev reward: %s" % (str(i + 1), reward_count), end="")

        # target al dan niet updaten
        targetupdatecounter += 1
        if targetupdatecounter == targetupdatefreq:
            targetupdatecounte = 0
            deepq.UpdateTargetModel()

    print()
    print(pgn)
