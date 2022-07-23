import unittest
from puzzle_generation import grid_puzzle, moving_puzzle
import tqdm
import cv2


class MovingPuzzleTest(unittest.TestCase):
    def test_sanity(self):
        total_steps = 300
        puzzle = grid_puzzle.image_to_grid_puzzle(image_path='../face.jpg', grid_size=16, puzzle_size=(1024, 1024), puzzle_pad=(128, 128))
        moving = moving_puzzle.create_random_moving_puzzle(puzzle, total_steps=total_steps)

        # animate the moving puzzle
        for i in tqdm.tqdm(range(total_steps)):
            moving.set_current_step(i)
            img, mask = moving.draw()
            cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
