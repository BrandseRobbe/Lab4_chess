from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent
from project.chess_engines.uci_engine import UciEngine

if __name__ == "__main__":
    utility = BoardUtility()
    agent = ChessAgent(utility, 14)
    # Create the engine
    engine = UciEngine("Custom chess engine", "Robbe, Kirsten & Ignace", agent)
    # Run the engine (will loop until the game is done or exited)
    engine.engine_operation()
