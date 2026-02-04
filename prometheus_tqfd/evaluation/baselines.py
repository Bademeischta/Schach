import random
import chess

class RandomPlayer:
    """Wählt uniform zufällig aus legalen Zügen"""
    def select_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))

class HeuristicPlayer:
    """
    Einfache regelbasierte Heuristik:
    - Materialbewertung
    - Zentrumskontrolle
    - Mobilität
    - Königssicherheit
    """

    PIECE_VALUES = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
    }

    CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

    def select_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None

        # 10% Exploration
        if random.random() < 0.1:
            return random.choice(legal_moves)

        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            board.push(move)
            score = self._evaluate(board)
            board.pop()

            # Perspektive: Nach unserem Zug ist Gegner dran
            # evaluate gibt wert für Weiß zurück. Wenn wir Schwarz sind, wollen wir niedrigen Wert.
            # Aber hier machen wir es einfacher: _evaluate gibt wert für Spieler am Zug zurück?
            # Nein, _evaluate gibt absoluten Wert (Weiß positiv).
            # Wenn wir am Zug sind und Weiß sind, wollen wir max score.
            # Wenn wir am Zug sind und Schwarz sind, wollen wir min score.

            actual_score = score if board.turn == chess.WHITE else -score

            if actual_score > best_score:
                best_score = actual_score
                best_move = move

        return best_move or random.choice(legal_moves)

    def _evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0

        # Material
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value

        # Zentrumskontrolle
        for sq in self.CENTER_SQUARES:
            if board.is_attacked_by(chess.WHITE, sq):
                score += 10
            if board.is_attacked_by(chess.BLACK, sq):
                score -= 10

        # Mobilität (approximiert durch legal moves)
        # Vorsicht: board.turn ändern ist gefährlich während iteration
        original_turn = board.turn
        board.turn = chess.WHITE
        score += len(list(board.legal_moves)) * 2
        board.turn = chess.BLACK
        score -= len(list(board.legal_moves)) * 2
        board.turn = original_turn

        return score
