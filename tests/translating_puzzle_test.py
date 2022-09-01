import unittest

import numpy as np
import matplotlib.pyplot as plt

from puzzle_generation import grid_puzzle, moving_puzzle
import tqdm
import cv2


class TranslatingPuzzleTest(unittest.TestCase):
    def test_sanity(self):
        total_steps = 100
        puzzle = grid_puzzle.image_to_grid_puzzle(image_path='../face.jpg', grid_size=16, puzzle_size=(1024, 1024), puzzle_pad=(128, 128))
        moving = moving_puzzle.create_random_translating_puzzle(puzzle, total_steps=total_steps)

        out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1024 + 128, 1024 + 128))
        out2 = cv2.VideoWriter('mask.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1024 + 128, 1024 + 128))


        # animate the moving puzzle
        for i in tqdm.tqdm(range(total_steps)):
            moving.set_current_step(i)
            img, mask = moving.draw()
            cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def test_shifting(self):
        total_steps = 100
        puzzle = grid_puzzle.image_to_grid_puzzle(image_path='../face.jpg', grid_size=16, puzzle_size=(1024, 1024), puzzle_pad=(128, 128))
        moving = moving_puzzle.create_random_shifting_puzzle(puzzle, total_steps=total_steps)

        # animate the moving puzzle
        for i in tqdm.tqdm(range(total_steps)):
            moving.set_current_step(i)
            img, mask = moving.draw()
            cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def test_vector_field_full(self):
        total_steps = 300
        puzzle = grid_puzzle.image_to_grid_puzzle(image_path='../face.jpg', grid_size=16, puzzle_size=(1024, 1024),
                                                  puzzle_pad=(128, 128))
        moving = moving_puzzle.create_random_translating_puzzle(puzzle, total_steps=total_steps)

        # animate the moving puzzle
        for i in tqdm.tqdm(range(total_steps)):
            moving.set_current_step(i)
            img, mask, vector_field = moving.draw_with_vector_field()
            #vector_field = np.concatenate([vector_field, np.zeros([vector_field.shape[0], vector_field.shape[1], 1])], axis=2)

            cv2.imshow('frame', np.linalg.norm(vector_field, axis=2))
            assert(vector_field.max() < 10) # if not, some piece is moving really fast

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()
