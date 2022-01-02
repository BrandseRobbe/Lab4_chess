import os
from collections import deque
import random
from datetime import datetime

import chess
import numpy as np

from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent, QLearning
from project.Custom_Agent.Neural_net import create_utilitymodel

epochs = 10000
batchsize = 32
learning_rate = 0.1  # kan nog gradueel verlaagt worden
discount = 0.9
epsilon = 0.8
decay = 0.999

winreward = 1000000
drawreward = 100
stepreward = -1

utility = BoardUtility()

agent = ChessAgent(utility, 1)

policyModel = create_utilitymodel()
deepq = QLearning(policyModel, batchsize, learning_rate, discount, epsilon, decay, winreward, drawreward, stepreward)

training_dataset = []

trainmodelfreq = 20
trainmodelcounter = 0

savemodelfreq = 200
savemodelcounter = 0
reward_count = [0, 0, 0]

if not os.path.exists("model_saves"):
    raise ModuleNotFoundError("Wrong working directory.")
working_dir = "model_saves/Session_%s" % len(os.listdir("model_saves"))
os.mkdir(working_dir)

for i in range(epochs):
    board = chess.Board()
    boardValue = BoardUtility.one_hot_board(board)
    pgn = ""
    for t in range(300):
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            break
        # whites move
        move, reward, done = deepq.GetAction(board)
        board.push(move)
        pgn += "%s. %s " % (t+1, move.uci())

        # startstate board toevoegen in one_hot vorm, next state in chess.board vorm omdat er nog extra bewerkingen nodig zijn.
        deepq.AddToMemory(boardValue, reward, board, done)
        boardValue = BoardUtility.one_hot_board(board)

        trainmodelcounter += 1
        if trainmodelfreq == trainmodelcounter:
            trainmodelcounter = 0
            deepq.UpdatePolicyModel()
            print("\rTraining episode: %s, prev reward: %s" % (str(i + 1), deepq.rewardcount), end="")

        # target al dan niet updaten
        savemodelcounter += 1
        if savemodelcounter == savemodelfreq:
            savemodelcounter = 0
            file_name = str(datetime.now().time().replace(microsecond=0)).replace(":", "_")
            path = working_dir + "/" + file_name + ".h5"
            print("\nsaved at %s" % (path))
            deepq.policyModel.save(path)
    print()
    print(pgn)
