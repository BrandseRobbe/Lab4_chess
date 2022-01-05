import os
from collections import deque
import random
from datetime import datetime

import chess
import numpy as np

from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent, QLearning
from project.Custom_Agent.Neural_net import create_utilitymodel

# Training hyperparameters
epochs = 10000
batchsize = 32
learning_rate = 0.1  # kan nog gradueel verlaagt worden
discount = 0.9
epsilon = 0.8
decay = 0.999
max_moves = 300

# Rewards
winreward = 1000000
drawreward = 100
stepreward = -1

# Objects
utility = BoardUtility()
agent = ChessAgent(utility, 1)
policyModel = create_utilitymodel()
deepq = QLearning(policyModel, batchsize, learning_rate,
                  discount, epsilon, decay, winreward, drawreward, stepreward)

# Training
training_dataset = []

trainmodelfreq = 20
trainmodelcounter = 0

# Model saving
savemodelfreq = 200
savemodelcounter = 0
reward_count = [0, 0, 0]


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

    # Next we determen the value of all the pieces on the board and pass this as a state
    boardValue = utility.one_hot_board(board)
    pgn = ""

    done = False

    # To make sure the game doesn't run endlessly, we capped the number of move that can be made in a game
    for t in range(max_moves):

        # In case that the board is in a checkmate or stalemate position OR one of the sides has insufficient training material, the training is terminated
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            break

        # Whites move
        white_move, white_reward, done = deepq.GetAction(board)
        board.push(move)
        pgn += f"{t+1}. {move.uci()} "

        """
        Saves the memories correctly:

        In order to do so, we have to calculate the next state for each side.
        This can only be done after the opposite site did their move.

        Fixing needed

        """

        # start state board toevoegen in one_hot vorm, next state in chess.board vorm omdat er nog extra bewerkingen nodig zijn.
        deepq.AddToMemory(boardValue, reward, board, done)
        boardValue = utility.one_hot_board(board)

        # Blacks move
        move, reward, done = deepq.GetAction(board)
        board.push(move)

        # start state board toevoegen in one_hot vorm, next state in chess.board vorm omdat er nog extra bewerkingen nodig zijn.
        deepq.AddToMemory(boardValue, reward, board, done)
        boardValue = utility.one_hot_board(board)

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
