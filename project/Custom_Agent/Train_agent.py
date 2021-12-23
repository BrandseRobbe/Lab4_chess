from collections import deque
import random

import chess
import numpy as np

from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent

epochs = 10000
batches = 32

utility = BoardUtility()

agent = ChessAgent(utility, 1)
epsilon = 0.8

training_dataset = []

for i in range(epochs):
    board = chess.Board()
    epsilon *= 0.9
    while True:
        if board.is_checkmate() or board.is_stalemate():
            break
        # whites move
        move, utility_val, board = agent.train_move(board)
        board.push(move)
        if board.is_checkmate():
            reward = 1000000
        elif board.is_stalemate():
            reward = 100
        else:
            reward = -1
        training_dataset.append((move, utility_val, board))



# voorbeeldcode
class QLearning():
    def __init__(self, policyModel, targetModel, possibleActions, batchsize, discount, epsilon, decay, sarsa=False):
        self.memory = deque(maxlen=5000) # deque gebruiken, enkel de laatste 5000 zetten onthouden

        self.batchsize = batchsize
        self.discount = discount
        self.epsilon = epsilon
        self.decay = decay

        self.policyModel = policyModel
        self.targetModel = targetModel

        self.possibleActions = possibleActions

    def AddToMemory(self, state, reward, new_state, done):
        self.memory.append((state, reward, new_state, done))

    def GetQValPolicy(self, board: chess.Board):
        # board omzetten naar iets dat in het model past
        # -> door policyModel halen
        state = ...
        qvalue = self.policyModel.predict(state)[0]
        return qvalue

    # Action with epsilon
    def GetAction(self, moves):
        qvals = [self.GetQValPolicy(moves) for _ in moves]
        # altijd blijven exploreren
        if self.epsilon > 0.1:
            self.epsilon *= self.decay
        if self.epsilon > random.random():
            return random.choice(self.possibleActions)
        else:
            return np.argmax(qvals)

    def UpdateTargetModel(self):
        self.targetModel.set_weights(self.policyModel.get_weights())

    def UpdatePolicyModel(self):
        if len(self.memory) < self.batchsize:
            return

        samples = random.sample(self.memory, self.batchsize)

        checkmate = np.asarray(list(zip(*samples))[4], dtype=bool)
        rewards = np.asarray(list(zip(*samples))[2], dtype=float)

        current_states = np.asarray(list(zip(*samples))[0], dtype=float)
        next_states = np.asarray(list(zip(*samples))[3], dtype=float)

        y = self.policyModel.predict(current_states)
        next_state_q_values = self.targetModel.predict(next_states)
        max_q_next_state = np.max(next_state_q_values, axis=1)

        for t in range(self.batchsize):
            if not checkmate[t]:
                y[t] = rewards[t] + self.discount * max_q_next_state[t]
            else:
                y[t] = rewards[t]  # als het schaakmat is, dan weten we de utility al.

        self.policyModel.fit(current_states, y, batch_size=self.batchsize, verbose=0)
