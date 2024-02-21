import chess_ai
import chess_ai.engine
import subprocess


def setup_engine(path):
    engine = chess_ai.engine.SimpleEngine.popen_uci(path)
    return engine


def play_game(player_engine, opponent_engine):
    board = chess_ai.Board()
    limit = chess_ai.engine.Limit(time=0.1)  # Adjust as necessary

    while not board.is_game_over():
        if board.turn == chess_ai.WHITE:
            result = player_engine.play(board, limit)
        else:
            result = opponent_engine.play(board, limit)
        board.push(result.move)

    print("Game Over. Result:", board.result())


if __name__ == "__main__":
    player_engine_path = "path/to/your/engine"  # Your neural network-based engine
    opponent_engine_path = "path/to/opponent/engine"  # Another UCI engine, e.g., Stockfish

    player_engine = setup_engine(player_engine_path)
    opponent_engine = setup_engine(opponent_engine_path)

    play_game(player_engine, opponent_engine)

    player_engine.quit()
    opponent_engine.quit()