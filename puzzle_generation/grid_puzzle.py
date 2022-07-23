from PIL import Image
import numpy as np
from typing import Tuple
import puzzle


def image_to_grid_puzzle(image_path:str, grid_size:int, puzzle_size: Tuple[int, int]=None, puzzle_pad: Tuple[int, int]=(0,0)):
    image = np.array(Image.open(image_path))
    if puzzle_size is None:
        puzzle_size = image.shape[:2]
    assert puzzle_size[0] >= image.shape[0] and puzzle_size[1] >= image.shape[1], "Can't create a puzzle that is" \
                                                                                  "smaller than the original image"

    padding = (puzzle_size[0] - image.shape[0]) // 2, (puzzle_size[1] - image.shape[1]) // 2
    created_puzzle = puzzle.Puzzle(image_size=puzzle_size, pad=puzzle_pad)
    tile_width, tile_height = image.shape[0] // grid_size, image.shape[1] // grid_size
    piece_count = 1
    for x in range(0, grid_size * tile_width, tile_width):
        for y in range(0, grid_size * tile_height, tile_height):
            current_tile = image[x: x + tile_width, y: y+tile_width]
            piece_location = x + padding[0], y + padding[1]
            current_piece = puzzle.Piece(piece_location[0], piece_location[1], theta=0, img=current_tile, mask_number=piece_count)
            piece_count += 1
            created_puzzle.add_piece(current_piece)

    return created_puzzle
