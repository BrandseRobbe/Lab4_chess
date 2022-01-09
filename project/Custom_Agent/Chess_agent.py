from collections import deque

import numpy as np

from project.Custom_Agent.Board_utility import BoardUtility
from project.chess_agents.agent import Agent
import chess
import chess.polyglot
from project.Custom_Agent.Board_utility import BoardUtility
import time
import random

"""An example search agent with two implemented methods to determine the next move"""


class ChessAgent():
    # Initialize your agent with whatever parameters you want
    def __init__(self, utility, time_limit=14.5) -> None:
        self.time_limit = time_limit
        self.opening = True
        self.utility = utility

    def calculate_move(self, board: chess.Board):
        start_time = time.time()
        # During the opening, use a chess move book to select moves
        if self.opening:
            with chess.polyglot.open_reader("opening_book/Perfect2017.bin") as reader:
                entries = list(reader.find_all(board))
                if len(entries) == 0:
                    self.opening = False
                else:
                    return entries[0].move

        # look if checkmate in 2 moves is guaranteed
        checkmate_move = BoardUtility.mate_in_x(board, 2)
        if checkmate_move is not None:
            return checkmate_move

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
                return move

            # Determine the value of the board after this move
            value = flip_value * self.utility.board_value(board)
            # If this is better than all other previous moves, store this move and its utility
            if value > highest_utility:
                best_move = move
                highest_utility = value

            board.pop()  # Revert the board to its original state, so we can try the next possible move

        return best_move


# voorbeeldcode
class QLearning():
    def __init__(self, policyModel, batchsize, learning_rate, discount, epsilon, decay, winreward, drawreward,
                 stepreward):
        # deque gebruiken, enkel de laatste 5000 zetten onthouden
        self.memory = deque(maxlen=5000)

        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.decay = decay

        self.winreward = winreward
        self.drawreward = drawreward
        self.stepreward = stepreward
        self.rewardcount = [0, 0, 0]

        self.policyModel = policyModel

    def AddToMemory(self, state, reward, new_state, done):
        self.memory.append((state, reward, new_state, done))

    def GetQValPolicy(self, board: chess.Board):
        # Get a usable representation of the board state
        boardvalue = BoardUtility.get_board_data(board)
        # predict data has to be slightly changed so dimentions fit for single prediction
        x1 = boardvalue[0][np.newaxis, ...]
        x2 = boardvalue[1][np.newaxis, ...]
        # Predict the qvalue for this state
        qvalue = self.policyModel.predict({"board_data": x1, "feature_data": x2})[0][0]
        return qvalue

    def rewardFunction(self, board: chess.Board, move: chess.Move):
        # Check which side is playing
        flip_value = 1 if board.turn == chess.WHITE else -1

        # Execute the move and calculate the utility of the new state
        board.push(move)
        utility = flip_value * self.GetQValPolicy(board)

        # Assign rewards according to the new board state
        if board.is_checkmate():
            reward = self.winreward
            done = True
        # board.is_seventyfive_moves() #board.is_fivefold_repetition() 2 more exit condidtions
        elif board.is_stalemate() or board.is_insufficient_material():
            reward = self.drawreward
            done = True
        else:
            reward = self.stepreward
            done = False
        # Undo the move and return the values
        board.pop()
        # black needs opposite rewards
        reward *= flip_value
        return reward, utility, done

    # Action with epsilon
    def GetAction(self, board: chess.Board):
        flip_value = 1 if board.turn == chess.WHITE else -1
        # look if checkmate in 2 moves is guaranteed to speed up learning
        checkmate_move = BoardUtility.mate_in_x(board, 2)
        if checkmate_move is not None:
            reward = flip_value * self.winreward
            board.push(checkmate_move)
            done = False
            if board.is_checkmate():
                done = True
                self.rewardcount[0] += 1

            board.pop()
            return checkmate_move, reward, done

        legal_moves = list(board.legal_moves)

        # Select a random move
        move = legal_moves.pop(random.randrange(len(legal_moves)))
        reward, highest_util, done = self.rewardFunction(board, move)

        # Determine wheter to take a random action or the best option based on a given epsilon
        if random.uniform(0, 1) < max([self.epsilon, 0.1]):
            self.epsilon *= self.decay
        else:
            # Loop trough all legal moves and select the best one
            for m in legal_moves:
                r, util, d = self.rewardFunction(board, m)

                if (flip_value == -1 and util < highest_util) or (flip_value == 1 and util > highest_util):
                    move = m
                    highest_util = util
                    reward = r
                    done = d

        # Keep track of the reeceived rewards
        if reward == self.stepreward or reward == -self.stepreward:
            self.rewardcount[2] += 1
        elif reward == self.winreward or reward == -self.winreward:
            self.rewardcount[0] += 1
        else:
            self.rewardcount[1] += 1

        return move, reward, done

    def UpdatePolicyModel(self):
        if len(self.memory) < self.batchsize:
            return

        samples = random.sample(self.memory, self.batchsize)

        current_states = np.asarray(list(zip(*samples))[0], dtype=object)
        rewards = np.asarray(list(zip(*samples))[1], dtype=float)
        next_states = np.asarray(list(zip(*samples))[2], dtype=object)
        done = np.asarray(list(zip(*samples))[3], dtype=bool)
        # next states bevat momenteel enkel nog maar de volgende state na een zet uit te voeren
        # het moet eigenlijk alle mogelijke volgende states bevatten zodat de max utility uit allemaal kan gehaald worden.

        one_hot = np.asarray(list(zip(*current_states))[0], dtype=float)
        features = np.asarray(list(zip(*current_states))[1], dtype=float)
        y = self.policyModel.predict({"board_data": one_hot, "feature_data": features})[..., 0]
        one_hot2 = np.asarray(list(zip(*current_states))[0], dtype=float)
        features2 = np.asarray(list(zip(*current_states))[1], dtype=float)
        y2 = self.policyModel.predict({"board_data": one_hot2, "feature_data": features2})[..., 0]

        for t in range(self.batchsize):
            if not done[t]:
                # werken met learning rate t.o.v. transition model (temporal-difference q-learning)
                # -> iets simpeler in uitvoering en ook een consistent result
                y[t] = y[t] + (self.learning_rate * (rewards[t] + (self.discount * y2[t]) - y[t]))
            else:
                # als het schaakmat is, dan weten we de utility al.
                y[t] = rewards[t]

                self.policyModel.fit({"board_data": one_hot, "feature_data": features}, y, batch_size=self.batchsize, verbose=0)
