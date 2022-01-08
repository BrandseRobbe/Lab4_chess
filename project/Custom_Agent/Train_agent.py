import os
import time
from collections import deque
import random
from datetime import datetime
import chess.polyglot

import chess
import numpy as np
from tqdm import tqdm

from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent, QLearning
from project.Custom_Agent.Neural_net import create_utilitymodel

# Training hyperparameters
epochs = 10000
batchsize = 64
learning_rate = 0.1  # kan nog gradueel verlaagt worden
discount = 0.9
epsilon = 0.8
decay = 0.999
max_moves = 300

# Rewards
winreward = 1000
drawreward = 10
stepreward = -1

# Objects
policyModel = create_utilitymodel()
deepq = QLearning(policyModel, batchsize, learning_rate,
                  discount, epsilon, decay, winreward, drawreward, stepreward)

# Training
training_dataset = []

trainmodelfreq = 5
trainmodelcounter = 0

# Model saving
savemodelfreq = 200
savemodelcounter = 0

# Creating a folder to save the trained model
if not os.path.exists("model_saves"):
    raise ModuleNotFoundError("Wrong working directory.")
working_dir = f"model_saves/Session_{len(os.listdir('model_saves'))}"
os.mkdir(working_dir)

# Start training
for i in range(epochs):

    # One epoch is equal to 1 game of chess
    # We start with creating a board for this game
    board = chess.Board()
    done = False

    # To make sure the game doesn't run endlessly, we capped the number of move that can be made in a game
    start_state = BoardUtility.get_board_data(board)
    opening = True
    pgn = ""
    for t in tqdm(range(max_moves)):
        # In case that the board is in a checkmate or stalemate position OR one of the sides has insufficient training material, the training is terminated
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            break

        # During the opening, use a chess move book to select moves
        if opening:
            with chess.polyglot.open_reader("opening_book/Perfect2017.bin") as reader:
                entries = list(reader.find_all(board))
                if len(entries) == 0:
                    opening = False
                    move, reward, done = deepq.GetAction(board)
                else:
                    # choose one out of the first 5 moves at random
                    # move = entries[min(random.randrange(0, 5), len(entries))].move
                    move = entries[0].move
                    reward = 0
                    done = False
        else:
            move, reward, done = deepq.GetAction(board)
        pgn += "%s. %s " % (t + 1, move.uci())
        board.push(move)
        new_state = BoardUtility.get_board_data(board)
        deepq.AddToMemory(start_state, reward, new_state, done)
        start_state = new_state

        # Terminate game when done
        if done:
            break

        trainmodelcounter += 1
        # Update the policy model
        if trainmodelfreq == trainmodelcounter:
            trainmodelcounter = 0
            deepq.UpdatePolicyModel()
            print("\rTraining episode: %s, prev reward: %s" %
                  (str(i + 1), deepq.rewardcount), end="")

        # Save the model
        savemodelcounter += 1
        if savemodelcounter == savemodelfreq:
            savemodelcounter = 0
            file_name = str(datetime.now().time().replace(
                microsecond=0)).replace(":", "_")
            path = working_dir + "/" + file_name + ".h5"
            print("\nsaved at %s" % (path))
            deepq.policyModel.save(path)

    print()
    print(pgn)
