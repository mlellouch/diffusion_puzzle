import unittest
from puzzle_generation import grid_puzzle
import matplotlib.pyplot as plt


class GridPuzzleCase(unittest.TestCase):
    def test_sanity(self):
        puzzle = grid_puzzle.image_to_grid_puzzle('../face.jpg', 16, puzzle_size=(1500, 1500))
        image, mask = puzzle.draw()
        plt.imshow(image)
        plt.show()
        plt.imshow(mask)
        plt.show()

    def test_turned(self):
        puzzle = grid_puzzle.image_to_grid_puzzle('../face.jpg', 16, puzzle_size=(1500, 1500))
        for p in puzzle.pieces:
            p.theta = 30
        image, mask = puzzle.draw()
        plt.imshow(image)
        plt.show()
        plt.imshow(mask)
        plt.show()



if __name__ == '__main__':
    unittest.main()
