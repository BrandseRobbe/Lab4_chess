from collections import deque

import numpy as np

from project.chess_agents.agent import Agent
import chess
from project.Custom_Agent import Board_utility
import time
import random

"""An example search agent with two implemented methods to determine the next move"""


class ChessAgent(Agent):
    # Initialize your agent with whatever parameters you want
    def __init__(self, utility, time_limit=14.5) -> None:
        super().__init__(utility=utility, time_limit_move=time_limit)
        self.time_limit = time_limit

    def train_move(self, board: chess.Board, epsilon: float):
        flip_value = 1 if board.turn == chess.WHITE else -1
        legal_moves = list(board.legal_moves)
        best_move = legal_moves.pop()  # al een random move nemen
        highest_utility = flip_value * self.utility.board_value(board)

        if random.random() < epsilon:
            return best_move

        # Loop trough all legal moves
        for move in legal_moves:
            board.push(move)  # Play the move
            if board.is_checkmate():
                best_move = move
                highest_utility = 1000000
                break

            # Determine the value of the board after this move
            value = flip_value * self.utility.board_value(board)
            if value > highest_utility:
                best_move = move
                highest_utility = value

        return best_move, flip_value * highest_utility, board

    def calculate_move(self, board: chess.Board):
        start_time = time.time()
        # If the agent is playing as black, the utility values are flipped (negative-positive)
        flip_value = 1 if board.turn == chess.WHITE else -1

        legal_moves = list(board.legal_moves)
        best_move = legal_moves.pop()  # al een random move nemen
        highest_utility = self.utility.board_value(board)
        # Loop trough all legal moves
        for move in legal_moves:
            # Check if the maximum calculation time for this move has been reached
            if time.time() - start_time > self.time_limit:
                break
            board.push(move)  # Play the move
            if board.is_checkmate():
                best_move = move
                highest_utility = 1000000
                break
            # get usefull info from position
            # board.is_stalemate()
            # board.is_insufficient_material()
            # board.outcome()
            # board.can_claim_threefold_repetition()
            # board.can_claim_draw()
            # castle_rights

            # Determine the value of the board after this move
            value = flip_value * ChessAgent.get_board_utility(board)
            # If this is better than all other previous moves, store this move and its utility
            if value > highest_utility:
                best_move = move
                highest_utility = value

            board.pop()  # Revert the board to its original state, so we can try the next possible move
        return best_move

    @property
    def color(self):
        return self.color

    @color.setter
    def color(self, value):
        if value in ["White", "Black"]:
            self.color = value
        else:
            raise ValueError("Chess color has to be 'White' or 'Black'")


# voorbeeldcode
class QLearning():
    def __init__(self, policyModel, targetModel, batchsize, discount, epsilon, decay, sarsa=False):
        self.memory = deque(maxlen=5000)  # deque gebruiken, enkel de laatste 5000 zetten onthouden

        self.batchsize = batchsize
        self.discount = discount
        self.epsilon = epsilon
        self.decay = decay

        self.policyModel = policyModel
        self.targetModel = targetModel

    def AddToMemory(self, state, reward, new_state, done, color):
        self.memory.append((state, reward, new_state, done, color))

    def GetQValPolicy(self, board: chess.Board):
        boardvalue = Board_utility.BoardUtility.one_hot_board(board)
        qvalue = self.policyModel.predict(np.expand_dims(boardvalue, axis=0))[0]
        return qvalue, boardvalue

    # Action with epsilon
    def GetAction(self, board: chess.Board):
        flip_value = 1 if board.turn == chess.WHITE else -1
        legal_moves = list(board.legal_moves)
        best_move = legal_moves.pop()
        highest_utility, boardvalue = self.GetQValPolicy(board)
        highest_utility *= flip_value

        if random.uniform(0, 1) < max([self.epsilon, 0.1]):
            self.epsilon *= self.decay
        else:
            # Loop trough all legal moves
            for move in legal_moves:
                board.push(move)  # Play the move
                if board.is_checkmate():
                    best_move = move
                    boardvalue = Board_utility.BoardUtility.one_hot_board(board)
                    board.pop()
                    break
                # Determine the value of the board after this move
                value, temp_board = self.GetQValPolicy(board)
                value *= flip_value
                if value > highest_utility:
                    best_move = move
                    highest_utility = value
                    boardvalue = temp_board
                board.pop()
        return best_move, boardvalue, flip_value

    def UpdateTargetModel(self):
        self.targetModel.set_weights(self.policyModel.get_weights())

    def UpdatePolicyModel(self):
        if len(self.memory) < self.batchsize:
            return

        samples = random.sample(self.memory, self.batchsize)

        current_states = np.asarray(list(zip(*samples))[0], dtype=float)
        rewards = np.asarray(list(zip(*samples))[1], dtype=float)
        next_states = np.asarray(list(zip(*samples))[2], dtype=float)
        done = np.asarray(list(zip(*samples))[3], dtype=bool)
        colors = np.asarray(list(zip(*samples))[4], dtype=bool)
        # next states bevat momenteel enkel nog maar de volgende state na een zet uit te voeren
        # het moet eigenlijk alle mogelijke volgende states bevatten zodat de max utility uit allemaal kan gehaald worden.

        y = self.policyModel.predict(current_states)
        next_state_q_values = self.targetModel.predict(next_states)
        max_q_next_state = np.max(next_state_q_values, axis=1)

        for t in range(self.batchsize):
            if not done[t]:
                utility = colors[t] * max_q_next_state[t]
                y[t] = rewards[t] + self.discount * utility
            else:
                y[t] = rewards[t]  # als het schaakmat is, dan weten we de utility al.

        self.policyModel.fit(current_states, y, batch_size=self.batchsize, verbose=0)
