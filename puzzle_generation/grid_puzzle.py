from PIL import Image
import numpy as np
from typing import Tuple
import puzzle
import cv2
from skimage.util import random_noise


def image_to_grid_puzzle(image_path:str, grid_size:int, puzzle_size: Tuple[int, int]=None, puzzle_pad: Tuple[int, int]=(0,0), add_noise=False):
    image = np.array(Image.open(image_path))
    # add alpha channel if needed
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    if puzzle_size is None:
        puzzle_size = image.shape[:2]

    else:
        image = cv2.resize(image, puzzle_size)

    if add_noise:
        image = np.clip((image + np.random.normal(0, 50, image.shape)), 0, 255).astype('uint8')

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
