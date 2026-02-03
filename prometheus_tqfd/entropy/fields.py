import chess
import numpy as np
import torch
import torch.nn.functional as F

class FieldCalculator:
    def __init__(self, physics_config):
        self.pc = physics_config
        self.piece_energies = {
            chess.PAWN: self.pc.energy_pawn,
            chess.KNIGHT: self.pc.energy_knight,
            chess.BISHOP: self.pc.energy_bishop,
            chess.ROOK: self.pc.energy_rook,
            chess.QUEEN: self.pc.energy_queen,
            chess.KING: self.pc.energy_king
        }

    def compute_fields(self, board: chess.Board):
        # M_total: Masse/Energie-Feld
        m_total = self._compute_mass_field(board)
        # F_pressure: Mobilitätsfluss-Feld
        f_pressure = self._compute_mobility_field(board)
        # P: Druck-Feld
        p_field = self._compute_pressure_field(m_total, f_pressure)
        # Φ: Potentialfeld
        phi = self._compute_potential_field(m_total, p_field)

        return torch.stack([m_total, f_pressure, p_field, phi]) # [4, 8, 8]

    def _compute_mass_field(self, board: chess.Board):
        field = np.zeros((8, 8), dtype=np.float32)
        sigma = self.pc.diffusion_sigma

        for square, piece in board.piece_map().items():
            r0, f0 = divmod(square, 8)
            energy = self.piece_energies[piece.piece_type]
            if piece.color == chess.BLACK:
                energy = -energy

            # Gaussian distribution
            for r in range(8):
                for f in range(8):
                    dist_sq = (r - r0)**2 + (f - f0)**2
                    field[r, f] += energy * np.exp(-dist_sq / (2 * sigma**2))

        return torch.from_numpy(field)

    def _compute_mobility_field(self, board: chess.Board):
        f_white = np.zeros((8, 8), dtype=np.float32)
        f_black = np.zeros((8, 8), dtype=np.float32)

        # White mobility
        board_w = board.copy()
        board_w.turn = chess.WHITE
        for move in board_w.pseudo_legal_moves:
            r, f = divmod(move.to_square, 8)
            f_white[r, f] += 1

        # Black mobility
        board_b = board.copy()
        board_b.turn = chess.BLACK
        for move in board_b.pseudo_legal_moves:
            r, f = divmod(move.to_square, 8)
            f_black[r, f] += 1

        # Normalization and Pressure
        f_pressure = (f_white - f_black)
        # Sigmoid-like normalization
        f_pressure = 2.0 / (1.0 + np.exp(-0.5 * f_pressure)) - 1.0

        return torch.from_numpy(f_pressure)

    def _compute_pressure_field(self, m_total, f_pressure):
        # P(x,y) = Conv2d(stack[M_total, F_pressure], kernel=3, σ_smooth=1.0)
        # For simplicity, just a smoothed combination
        combined = torch.stack([m_total, f_pressure]).unsqueeze(0)
        kernel = torch.ones((1, 2, 3, 3)) / 18.0
        p_field = F.conv2d(combined, kernel, padding=1)
        return p_field.squeeze()

    def _compute_potential_field(self, m_total, p_field):
        # Φ(x,y) = α × M_total(x,y) + β × P(x,y) + γ × ∇²M(x,y)
        # ∇²M approximated with Laplacian kernel
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().view(1, 1, 3, 3)
        laplacian = F.conv2d(m_total.view(1, 1, 8, 8), laplacian_kernel, padding=1).squeeze()

        phi = (self.pc.field_alpha * m_total +
               self.pc.field_beta * p_field +
               self.pc.field_gamma * laplacian)
        return phi
