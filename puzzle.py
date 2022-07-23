from typing import Tuple, List

import numpy as np
import cv2


class Piece:

    def __init__(self, x, y, theta, img, mask_number:int =1):
        assert img.shape[2] == 4, "only accept images with alpha, so we know which pixels are part of the piece"
        self.x = x
        self.y = y
        self.theta = theta % 360
        self.img = img[:, :, :3]
        self.mask_number = mask_number
        self.mask = img[:, :, 3]
        self.mask[self.mask >= 1] = 1
        self.mask = self.mask.astype(np.uint16) * mask_number


    def rotate_image(self, image, theta):
        # Taking image height and width
        imgHeight, imgWidth = image.shape[0], image.shape[1]
        centreY, centreX = imgHeight // 2, imgWidth // 2
        rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), theta, 1.0)

        cosofRotationMatrix = np.abs(rotationMatrix[0][0])
        sinofRotationMatrix = np.abs(rotationMatrix[0][1])

        newImageHeight = int((imgHeight * sinofRotationMatrix) +
                             (imgWidth * cosofRotationMatrix))
        newImageWidth = int((imgHeight * cosofRotationMatrix) +
                            (imgWidth * sinofRotationMatrix))

        rotationMatrix[0][2] += (newImageWidth / 2) - centreX
        rotationMatrix[1][2] += (newImageHeight / 2) - centreY

        # Now, we will perform actual image rotation
        rotatingimage = cv2.warpAffine(
            image, rotationMatrix, (newImageWidth, newImageHeight))

        return rotatingimage

    def draw_piece(self):
        return self.x, self.y, self.rotate_image(self.img, self.theta), self.rotate_image(self.mask, self.theta)

    def __str__(self):
        return f'<{self.x} {self.y}>'


class Puzzle:
    image_size: Tuple[int, int]
    pieces: list[Piece]

    def __init__(self, image_size: Tuple[int, int], pad: Tuple[int, int]):
        self.pieces = []
        self.image_size = image_size
        self.pad = pad
        self.real_size = image_size[0] + pad[0], image_size[1] + pad[1]

    def add_piece(self, piece: Piece):
        self.pieces.append(piece)

    def draw(self):
        """
        Returns the way the puzzle looks, and a segmentation mask
        :param pad_image: Pad the image so that all the puzzles will fit in
        """

        canvas = np.zeros((self.real_size[0], self.real_size[1], 3), dtype=np.uint8)
        mask_canvas = np.zeros(self.real_size[:2], dtype=np.uint16)
        for p in self.pieces:
            x, y, img, mask = p.draw_piece()
            x, y = x + (self.pad[0] // 2),  y + (self.pad[1] // 2)
            canvas[x:x + img.shape[0], y:y + img.shape[1]][mask != 0] = img[mask != 0]
            mask_canvas[x:x + mask.shape[0], y:y + mask.shape[1]][mask != 0] = mask[mask != 0]

        return canvas, mask_canvas


    @property
    def number_of_pieces(self):
        return len(self.pieces)

    def save_puzzle(self):
        pass

    def load_puzzle(self):
        pass


class MovingPiece(Piece):

    def __init__(self, piece: Piece, target_x, target_y, target_theta, total_steps: int):
        # rebuild the piece alpha channel
        mask = piece.mask.copy()[:, :, np.newaxis]
        mask[mask >= 1] = 1
        img = np.concatenate([piece.img, mask.astype(np.uint8)], axis=2)
        super().__init__(piece.x, piece.y, piece.theta, img, piece.mask_number)
        self.target_x = target_x
        self.target_y = target_y
        self.target_theta = target_theta % 360
        self.total_steps = total_steps
        self.current_step = 0

    def set_step(self, current_step):
        self.current_step = current_step

    def _get_step_value(self, start, end):
        return start + (end - start) * (self.current_step / self.total_steps)

    def get_step_location(self):
        x = self._get_step_value(self.x, self.target_x)
        y = self._get_step_value(self.y, self.target_y)
        theta = self._get_step_value(self.theta, self.target_theta)
        return int(x), int(y), theta

    def draw_piece(self):
        x, y, theta = self.get_step_location()
        return x, y, self.rotate_image(self.img, theta), self.rotate_image(self.mask, theta)


class MovingPuzzle(Puzzle):
    pieces:List[MovingPiece]

    def __init__(self, image_size: Tuple[int, int], puzzle_pad: Tuple[int, int]=(0,0)):
        super().__init__(image_size, puzzle_pad)

    def set_current_step(self, current_step):
        for p in self.pieces:
            p.set_step(current_step)
