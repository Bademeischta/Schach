import chess
import numpy as np
import torch
from prometheus_tqfd.config import PrometheusConfig

class PhysicsFieldCalculator:
    """
    Berechnet physik-inspirierte Felder aus Board-Zustand.

    Ausgabe: Tensor[3, 8, 8]
    - Kanal 0: Masse-Feld M(x,y) - Gauß-gewichtete Figurenenergie
    - Kanal 1: Mobilitäts-Feld F(x,y) - Angriffs-/Bewegungsdruck
    - Kanal 2: Druck-Feld P(x,y) - Kombination aus M und F
    """

    def __init__(self, config: PrometheusConfig):
        self.energies = {
            chess.KING: config.physics_energy_king,
            chess.QUEEN: config.physics_energy_queen,
            chess.ROOK: config.physics_energy_rook,
            chess.BISHOP: config.physics_energy_bishop,
            chess.KNIGHT: config.physics_energy_knight,
            chess.PAWN: config.physics_energy_pawn,
        }
        self.sigma = config.physics_diffusion_sigma
        self.alpha = config.physics_field_alpha
        self.beta = config.physics_field_beta

        self._precompute_gaussian_kernel()

    def _precompute_gaussian_kernel(self):
        # 15x15 kernel um jede Position abzudecken
        size = 15
        center = size // 2
        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist_sq = (i - center)**2 + (j - center)**2
                kernel[i, j] = np.exp(-dist_sq / (2 * self.sigma**2))
        self.kernel = kernel

    def _gaussian_at(self, x0, y0):
        # Returns a 8x8 grid with a gaussian centered at x0, y0
        grid = np.zeros((8, 8))
        size = 15
        center = size // 2

        for r in range(8):
            for c in range(8):
                dr, dc = r - y0, c - x0
                if abs(dr) <= center and abs(dc) <= center:
                    grid[r, c] = self.kernel[center + dr, center + dc]
        return grid

    def compute(self, board: chess.Board) -> torch.Tensor:
        """
        Berechnet alle drei Felder.
        """
        mass = self._compute_mass_field(board)
        mobility = self._compute_mobility_field(board)

        # Druck-Feld: alpha*M + beta*F (geglättet)
        # Hier vereinfacht als direkte Kombination
        pressure = self.alpha * mass + self.beta * mobility

        # Stack and to tensor
        fields = np.stack([mass, mobility, pressure])
        return torch.from_numpy(fields).float()

    def _compute_mass_field(self, board: chess.Board) -> np.ndarray:
        field = np.zeros((8, 8))
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                energy = self.energies[piece.piece_type]
                sign = 1 if piece.color == chess.WHITE else -1
                x0, y0 = square % 8, square // 8
                field += sign * energy * self._gaussian_at(x0, y0)

        # Normalisierung
        if np.max(np.abs(field)) > 0:
            field = field / np.max(np.abs(field))
        return field

    def _compute_mobility_field(self, board: chess.Board) -> np.ndarray:
        field = np.zeros((8, 8))
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            for square in chess.SQUARES:
                attackers = len(board.attackers(color, square))
                x, y = square % 8, square // 8
                field[y, x] += sign * attackers

        # Normalisieren auf [-1, 1]
        if field.max() - field.min() > 0:
            field = 2 * (field - field.min()) / (field.max() - field.min()) - 1
        return field
