import chess
import random
import numpy as np

class RandomPlayer:
    def select_move(self, board):
        return random.choice(list(board.legal_moves))

class HeuristicPlayer:
    def __init__(self, exploration=0.1):
        self.exploration = exploration
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

    def select_move(self, board):
        if random.random() < self.exploration:
            return random.choice(list(board.legal_moves))

        legal_moves = list(board.legal_moves)
        best_score = -float('inf')
        best_moves = []

        for move in legal_moves:
            board.push(move)
            score = self.evaluate(board)
            board.pop()

            # Since evaluate returns score for White, and we want to maximize for the player whose turn it was
            if board.turn == chess.BLACK:
                score = -score

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def evaluate(self, board):
        if board.is_checkmate():
            return -9999 if board.turn == chess.WHITE else 9999

        score = 0
        # Material
        for square, piece in board.piece_map().items():
            val = self.piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += val * 10
            else:
                score -= val * 10

        # Mobility
        score += board.legal_moves.count() * (1 if board.turn == chess.WHITE else -1)

        # Center control (d4, e4, d5, e5)
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for sq in center_squares:
            if board.is_attacking(chess.WHITE, sq): score += 0.5
            if board.is_attacking(chess.BLACK, sq): score -= 0.5

        return score
