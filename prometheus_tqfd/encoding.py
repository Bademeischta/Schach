import chess
import numpy as np
import torch

class BoardEncoder:
    def __init__(self, use_history=False, history_len=8):
        self.use_history = use_history
        self.history_len = history_len
        self.num_channels = 19 + (12 * (history_len - 1)) if use_history else 19

    def encode(self, board: chess.Board) -> torch.Tensor:
        # [C, 8, 8]
        tensor = np.zeros((19, 8, 8), dtype=np.float32)

        # Piece channels
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            # White: 0-5, Black: 6-11
            channel = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
            tensor[channel, rank, file] = 1.0

        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE): tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0

        # En passant
        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            tensor[16, rank, file] = 1.0

        # Halfmove clock
        tensor[17, :, :] = board.halfmove_clock / 100.0

        # Side to move
        if board.turn == chess.WHITE:
            tensor[18, :, :] = 1.0

        return torch.from_numpy(tensor)

class MoveEncoder:
    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._create_mapping()

    def _create_mapping(self):
        idx = 0
        # directions: (dr, df)
        # Queen moves
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]

        for from_sq in range(64):
            from_rank, from_file = divmod(from_sq, 8)

            # Queen-like moves
            for d_idx, (dr, df) in enumerate(directions):
                for dist in range(1, 8):
                    to_rank, to_file = from_rank + dr * dist, from_file + df * dist
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        # This move exists. We need a unique ID for (from_sq, d_idx, dist)
                        # Actually AlphaZero maps each move to 73 planes of 8x8
                        # 0-55: Queen moves (8 dir * 7 dist)
                        # 56-63: Knight moves
                        # 64-72: Underpromotions
                        pass

        # Let's use a simpler mapping that's consistent with the spec: 64 * 73
        # plane_idx:
        # 0..55: Queen moves (direction * 7 + (distance - 1))
        # 56..63: Knight moves
        # 64..72: Underpromotions

        # We don't really need to pre-populate everything if we can compute it
        pass

    def get_plane_and_sq(self, move: chess.Move):
        from_sq = move.from_square
        to_sq = move.to_square
        from_rank, from_file = divmod(from_sq, 8)
        to_rank, to_file = divmod(to_sq, 8)
        dr, df = to_rank - from_rank, to_file - from_file

        # Underpromotions
        if move.promotion and move.promotion != chess.QUEEN:
            # 64-72: 3 directions (df: -1, 0, 1) x 3 pieces (N, B, R)
            # Pieces: N=2, B=3, R=4
            piece_idx = move.promotion - 2 # 0, 1, 2
            dir_idx = df + 1 # 0, 1, 2
            plane = 64 + piece_idx * 3 + dir_idx
            return plane, from_sq

        # Knight moves
        knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        if (dr, df) in knight_moves:
            plane = 56 + knight_moves.index((dr, df))
            return plane, from_sq

        # Queen moves
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        abs_dr, abs_df = abs(dr), abs(df)
        dist = max(abs_dr, abs_df)
        if (dr // dist, df // dist) in directions and (abs_dr == 0 or abs_df == 0 or abs_dr == abs_df):
            d_idx = directions.index((dr // dist, df // dist))
            plane = d_idx * 7 + (dist - 1)
            return plane, from_sq

        raise ValueError(f"Invalid move for encoding: {move}")

    def move_to_index(self, move: chess.Move) -> int:
        plane, from_sq = self.get_plane_and_sq(move)
        return from_sq * 73 + plane

    def index_to_move(self, index: int, board: chess.Board) -> chess.Move:
        from_sq, plane = divmod(index, 73)
        from_rank, from_file = divmod(from_sq, 8)

        if plane < 56:
            # Queen move
            d_idx, dist_m1 = divmod(plane, 7)
            dist = dist_m1 + 1
            directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
            dr, df = directions[d_idx]
            to_rank, to_file = from_rank + dr * dist, from_file + df * dist
            to_sq = to_rank * 8 + to_file
            move = chess.Move(from_sq, to_sq)
            # Check for promotion to Queen (default)
            if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN:
                if (to_rank == 7 and board.turn == chess.WHITE) or (to_rank == 0 and board.turn == chess.BLACK):
                    move.promotion = chess.QUEEN
            return move
        elif plane < 64:
            # Knight move
            knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
            dr, df = knight_moves[plane - 56]
            to_rank, to_file = from_rank + dr, from_file + df
            to_sq = to_rank * 8 + to_file
            return chess.Move(from_sq, to_sq)
        else:
            # Underpromotion
            piece_idx, dir_idx = divmod(plane - 64, 3)
            df = dir_idx - 1
            piece = piece_idx + 2
            to_rank = 7 if board.turn == chess.WHITE else 0
            to_file = from_file + df
            to_sq = to_rank * 8 + to_file
            return chess.Move(from_sq, to_sq, promotion=piece)

    def get_legal_mask(self, board: chess.Board) -> torch.Tensor:
        mask = torch.zeros(4672, dtype=torch.bool)
        for move in board.legal_moves:
            mask[self.move_to_index(move)] = True
        return mask
