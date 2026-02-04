import chess
import torch
from typing import Dict, Any, List, Optional, Tuple
from prometheus_tqfd.encoding import MoveEncoder
from prometheus_tqfd.config import PrometheusConfig

class TacticsDetector:
    """
    Regelbasierter Detektor für kritische taktische Muster.
    KEINE ML - reine Schachlogik via python-chess.

    Erkennt:
    - Matt in 1 (für uns)
    - Matt-Drohung (vom Gegner)
    - Hängende Figuren
    - Verfügbare Schachs
    """

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.move_encoder = MoveEncoder()
        self.boost_strength = config.tactics_boost_strength
        self.decay_rate = config.tactics_boost_decay
        self.current_strength = self.boost_strength

    def decay_step(self):
        self.current_strength *= self.decay_rate

    def detect(self, board: chess.Board) -> Dict[str, Any]:
        threats = {
            'mate_in_1': None,          # Der Matt-Zug, wenn vorhanden
            'mate_threat': False,        # Gegner droht Matt
            'hanging_pieces': [],        # Ungeschützte eigene Figuren (square, piece)
            'checks_available': 0,       # Anzahl möglicher Schachs
            'captures_available': [],    # Schlagzüge
        }

        # Matt in 1 suchen
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                threats['mate_in_1'] = move
                board.pop()
                break
            board.pop()

        # Gegner-Matt-Drohung prüfen
        # Wir machen einen Nullzug um zu sehen ob der Gegner Matt setzen kann
        if not board.is_check():
            board.push(chess.Move.null())
            for move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    threats['mate_threat'] = True
                    board.pop()
                    break
                board.pop()
            board.pop()

        # Hängende Figuren finden
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    defenders = len(list(board.attackers(board.turn, square)))
                    attackers = len(list(board.attackers(not board.turn, square)))
                    if defenders < attackers:
                        threats['hanging_pieces'].append((square, piece))

        # Schachs zählen und Schlagzüge sammeln
        for move in board.legal_moves:
            if board.gives_check(move):
                threats['checks_available'] += 1
            if board.is_capture(move):
                threats['captures_available'].append(move)

        return threats

    def get_tactical_boost(self, board: chess.Board) -> torch.Tensor:
        """
        Erzeugt Boost-Tensor für taktisch kritische Züge.
        Wird zur Policy addiert (vor Softmax).
        """
        boost = torch.zeros(4672)
        threats = self.detect(board)

        # Matt in 1: Massiver Boost
        if threats['mate_in_1']:
            idx = self.move_encoder.move_to_index(threats['mate_in_1'])
            boost[idx] = self.config.tactics_mate_boost

        # Matt-Drohung: Defensive Züge boosten
        if threats['mate_threat']:
            for move in board.legal_moves:
                # Schach geben ist oft defensiv
                if board.gives_check(move):
                    idx = self.move_encoder.move_to_index(move)
                    boost[idx] += self.config.tactics_threat_boost
                # König bewegen
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.KING:
                    idx = self.move_encoder.move_to_index(move)
                    boost[idx] += self.config.tactics_threat_boost * 0.5

        # Hängende Figuren retten
        for square, piece in threats['hanging_pieces']:
            for move in board.legal_moves:
                if move.from_square == square:
                    idx = self.move_encoder.move_to_index(move)
                    value = self._piece_value(piece.piece_type)
                    boost[idx] += value * (self.config.tactics_hanging_boost / 10.0)

        return boost * self.current_strength

    def _piece_value(self, piece_type: int) -> float:
        values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
        }
        return values.get(piece_type, 0)
