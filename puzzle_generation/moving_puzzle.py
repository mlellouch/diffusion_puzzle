import puzzle
from typing import Tuple
import math
import random


def find_target_location_for_piece(piece:puzzle.Piece, canvas_size: Tuple[int, int]):
    # in the maximal case, the piece is rotated 45 degrees
    max_piece_size = int(piece.img.shape[0] * math.sqrt(2)), int(piece.img.shape[1] * math.sqrt(2))
    target_x = random.randint(max_piece_size[0], canvas_size[0] - max_piece_size[0])
    target_y = random.randint(max_piece_size[1], canvas_size[1] - max_piece_size[1])
    target_theta = random.random() * 360.0
    return target_x, target_y, target_theta


def create_random_moving_puzzle(original_puzzle:puzzle.Puzzle, total_steps:int):
    new_puzzle = puzzle.MovingPuzzle(original_puzzle.image_size, puzzle_pad=original_puzzle.pad)
    for piece in original_puzzle.pieces:
        target_x, target_y, target_theta = find_target_location_for_piece(piece, original_puzzle.image_size)
        new_piece = puzzle.MovingPiece(piece, target_x, target_y, target_theta, total_steps)
        new_puzzle.add_piece(new_piece)

    return new_puzzle


