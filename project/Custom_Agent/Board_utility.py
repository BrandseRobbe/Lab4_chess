import chess
from project.chess_utilities.utility import Utility


class BoardUtility(Utility):
    def __init__(self) -> None:
        pass

    # hier wordt board info door het neural network gehaald om de utility te berekenen
    def board_value(self, board: chess.Board):
        pass

