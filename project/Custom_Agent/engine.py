from project.Custom_Agent.Board_utility import BoardUtility
from project.Custom_Agent.Chess_agent import ChessAgent
from project.Custom_Agent.uci_engine import UciEngine

if __name__ == "__main__":
    model_file = "project/Custom_Agent/chess_model.h5"
    utility = BoardUtility()
    utility.load_chess_model(model_file)
    agent = ChessAgent(utility, 14)

    # Create the engine
    engine = UciEngine(name="Custom chess engine", author="Robbe, Kirsten & Ignace", agent=agent)
    # Run the engine (will loop until the game is done or exited)
    engine.engine_operation()
