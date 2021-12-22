from project.chess_agents.agent import Agent
import chess
from project.chess_utilities.utility import Utility
import time
import random

"""An example search agent with two implemented methods to determine the next move"""


class ChessAgent(Agent):
    # Initialize your agent with whatever parameters you want
    def __init__(self, color, time_limit=14.5):
        self.color = color
        self.time_limit = time_limit

    def get_board_utility(boardinfo: chess.Board) -> float:
        # hier wordt board info door het neural network gehaald om de utility te berekenen
        pass

    # This agent does not perform any searching, it sinmply iterates trough all the moves possible and picks the one with the highest utility
    def calculate_move(self, board: chess.Board):
        start_time = time.time()
        # If the agent is playing as black, the utility values are flipped (negative-positive)
        flip_value = 1 if self.color is "White" else -1

        best_move = random.sample(list(board.legal_moves), 1)[0]  # al een random move nemen
        highest_utility = 0
        # Loop trough all legal moves
        for move in list(board.legal_moves):
            # Check if the maximum calculation time for this move has been reached
            if time.time() - start_time > self.time_limit:
                break
            board.push(move)  # Play the move
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
