from project.chess_agents.agent import Agent
import chess
from project.chess_utilities.utility import Utility
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
