from typing import Tuple, List

import torch.nn
from torchvision.transforms import Compose, ToTensor, Normalize

import numpy as np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


    def get_rotation_matrix(self, image, theta):
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
        return rotationMatrix, newImageWidth, newImageHeight

    def rotate_image(self, image, theta):
        rotationMatrix, newImageWidth, newImageHeight = self.get_rotation_matrix(image, theta)

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
    pieces: List[Piece]

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


        x_space, y_space = np.linspace(0, self.img.shape[0] - 1, self.img.shape[0]), np.linspace(0, self.img.shape[1] - 1, self.img.shape[1])
        xs, ys = np.meshgrid(x_space, y_space)
        self.initial_pixel_position = np.stack([ys,xs], axis=2)
        # remove me
        # for vector field monitoring
        self.pixel_location = np.zeros_like(self.mask, dtype=np.uint16)
        pixels_to_follow = self.mask.sum() // self.mask_number
        assert pixels_to_follow < (2 ** 16) - 1, "for computation purposes, can't have a piece with 2**16 pixels"
        self.pixel_location[self.mask != 0] = np.arange(1, pixels_to_follow + 1)


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

    def draw_piece_with_vector_field(self):
        x, y, theta = self.get_step_location()
        vector_field = self.rotate_image(self.initial_pixel_position, theta) + np.array([x,y])
        return x, y, self.rotate_image(self.img, theta), self.rotate_image(self.mask, theta), vector_field


    def get_pixel_position(self):
        x, y, theta = self.get_step_location()

        # get current mat
        rotation_matrix, _, _ = self.get_rotation_matrix(self.img, theta)

        # perform the transform on all pixels, and get their new position
        new_position = np.einsum('lk,ijk->ijl', rotation_matrix, self.initial_pixel_position)
        new_position += np.array([x,y])
        return new_position


class MovingPuzzle(Puzzle):
    pieces:List[MovingPiece]

    def __init__(self, image_size: Tuple[int, int], puzzle_pad: Tuple[int, int]=(0,0)):
        super().__init__(image_size, puzzle_pad)
        self.last_pixel_locations = []

    def add_piece(self, piece: MovingPiece):
        super().add_piece(piece)
        self.last_pixel_locations.append(piece.initial_pixel_position[:, :, :2])

    def set_current_step(self, current_step):
        for p in self.pieces:
            p.set_step(current_step)

    def draw_with_vector_field(self):
        """
        Returns the way the puzzle looks, and a segmentation mask
        :param pad_image: Pad the image so that all the puzzles will fit in
        """

        canvas = np.zeros((self.real_size[0], self.real_size[1], 3), dtype=np.uint8)
        total_vector_field = np.zeros((self.real_size[0], self.real_size[1], 2), dtype=np.uint8)
        mask_canvas = np.zeros(self.real_size[:2], dtype=np.uint16)
        new_pixel_locations = []
        for p, last_pixel_location in zip(self.pieces, self.last_pixel_locations):
            x, y, img, mask, vector_field = p.draw_piece_with_vector_field()
            x, y = x + (self.pad[0] // 2),  y + (self.pad[1] // 2)
            canvas[x:x + img.shape[0], y:y + img.shape[1]][mask != 0] = img[mask != 0]
            mask_canvas[x:x + mask.shape[0], y:y + mask.shape[1]][mask != 0] = mask[mask != 0]

            d_vector_field = vector_field - last_pixel_location
            new_pixel_locations.append(vector_field)
            total_vector_field[x:x + img.shape[0], y:y + img.shape[1]][mask != 0] = d_vector_field[mask != 0]

        self.last_pixel_locations = new_pixel_locations
        return canvas, mask_canvas, total_vector_field


    def draw_vector_field(self):
        """
        Note: calling this function will update self.last_pixel_location
        hence, if this function is called twice without changing the current step,
        the second call should return all zeros
        """

        raise DeprecationWarning('this doesn\'t work')
        canvas = np.zeros((self.real_size[0], self.real_size[1], 2), dtype=np.uint8)
        for p, last_pixel_position in zip(self.pieces, self.last_pixel_locations):
            piece_vector_field = p.get_pixel_position() - last_pixel_position
            x, y, _, mask = p.draw_piece()
            x, y = x + (self.pad[0] // 2),  y + (self.pad[1] // 2)
            canvas[x:x + piece_vector_field.shape[0], y:y + piece_vector_field.shape[1]][mask != 0] = piece_vector_field[mask != 0]

        return canvas


### simple case where the pieces don't rotate
class TranslatingPiece(Piece):

    def __init__(self, piece: Piece, target_x, target_y, total_steps: int):
        # rebuild the piece alpha channel
        mask = piece.mask.copy()[:, :, np.newaxis]
        mask[mask >= 1] = 1
        img = np.concatenate([piece.img, mask.astype(np.uint8)], axis=2)
        super().__init__(piece.x, piece.y, theta=0, img=img, mask_number=piece.mask_number)
        self.target_x = target_x
        self.target_y = target_y
        self.total_steps = total_steps
        self.current_step = 0

        x_space, y_space = np.linspace(0, self.img.shape[0] - 1, self.img.shape[0]), np.linspace(0, self.img.shape[1] - 1, self.img.shape[1])
        xs, ys = np.meshgrid(x_space, y_space)
        self.initial_pixel_position = np.stack([ys,xs], axis=2)
        # remove me
        # for vector field monitoring
        self.pixel_location = np.zeros_like(self.mask, dtype=np.uint16)
        pixels_to_follow = self.mask.sum() // self.mask_number
        assert pixels_to_follow < (2 ** 16) - 1, "for computation purposes, can't have a piece with 2**16 pixels"
        self.pixel_location[self.mask != 0] = np.arange(1, pixels_to_follow + 1)


    def set_step(self, current_step):
        self.current_step = current_step

    def _get_step_value(self, start, end):
        return start + (end - start) * (self.current_step / self.total_steps)

    def get_step_location(self):
        x = self._get_step_value(self.x, self.target_x)
        y = self._get_step_value(self.y, self.target_y)
        return int(x), int(y), 0

    def draw_piece(self):
        x, y, _ = self.get_step_location()
        return x, y, self.img, self.mask

    def draw_piece_with_vector_field(self):
        x, y, _ = self.get_step_location()
        vector_field = self.initial_pixel_position + np.array([x,y])
        return x, y, self.img, self.mask, vector_field


class TranslatingPuzzle(Puzzle):
    pieces:List[TranslatingPiece]

    def __init__(self, image_size: Tuple[int, int], puzzle_pad: Tuple[int, int]=(0,0)):
        super().__init__(image_size, puzzle_pad)
        self.last_pixel_locations = []

    def add_piece(self, piece: TranslatingPiece):
        super().add_piece(piece)
        self.last_pixel_locations.append(piece.initial_pixel_position[:, :, :2] + np.array([piece.x, piece.y]))

    def set_current_step(self, current_step):
        for p in self.pieces:
            p.set_step(current_step)

    def draw_with_vector_field(self):
        """
        Returns the way the puzzle looks, and a segmentation mask
        :param pad_image: Pad the image so that all the puzzles will fit in
        """

        canvas = np.zeros((self.real_size[0], self.real_size[1], 3), dtype=np.uint8)
        total_vector_field = np.zeros((self.real_size[0], self.real_size[1], 3), dtype=np.float32)
        mask_canvas = np.zeros(self.real_size[:2], dtype=np.uint16)
        new_pixel_locations = []
        for p, last_pixel_location in zip(self.pieces, self.last_pixel_locations):
            x, y, img, mask, vector_field = p.draw_piece_with_vector_field()
            x, y = x + (self.pad[0] // 2),  y + (self.pad[1] // 2)
            canvas[x:x + img.shape[0], y:y + img.shape[1]][mask != 0] = img[mask != 0]
            mask_canvas[x:x + mask.shape[0], y:y + mask.shape[1]][mask != 0] = mask[mask != 0]

            d_vector_field = vector_field - last_pixel_location
            new_pixel_locations.append(vector_field)

            vector_field_to_update = d_vector_field[mask != 0]
            total_vector_field[x:x + img.shape[0], y:y + img.shape[1]][mask != 0] = np.concatenate([vector_field_to_update, np.zeros([vector_field_to_update.shape[0], 1])], axis=1)

        self.last_pixel_locations = new_pixel_locations
        return canvas, mask_canvas, total_vector_field



    def run_model_on_puzzle(self, model: torch.nn.Module, initial_step: int, resolve_step:int = 0):
        image_transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        field_transforms = Compose([
            ToTensor(),
            # Normalize((0,0,0), (0.005, 0.005, 0.005))
        ])

        self.set_current_step(initial_step)

        def draw_pieces(pieces):
            pad_size = pieces[0][2].shape[:2]
            canvas = np.zeros((self.real_size[0] + pad_size[0]*2, self.real_size[1]+ pad_size[1]*2, 3), dtype=np.uint8)
            mask_canvas = np.zeros(canvas.shape[:2], dtype=np.uint16)

            for p in pieces:
                x, y, img, mask = p
                x += pad_size[0]
                y += pad_size[1]
                x_in_canvas = 0 <= x < canvas.shape[0] and (x + img.shape[0] < canvas.shape[0])
                y_in_canvas = 0 <= y < canvas.shape[1] and (y + img.shape[1] < canvas.shape[1])

                if x_in_canvas and y_in_canvas:
                    canvas[x:x + img.shape[0], y:y + img.shape[1]][mask != 0] = img[mask != 0]
                    mask_canvas[x:x + mask.shape[0], y:y + mask.shape[1]][mask != 0] = mask[mask != 0]

            return canvas[pad_size[0]: -pad_size[0], pad_size[1]: -pad_size[1]], mask_canvas[pad_size[0]: -pad_size[0], pad_size[1]: -pad_size[1]]

        def move_pieces_according_to_field(field, mask_canvas, pieces):
            for mask_number in range(1, len(pieces)+1):
                vectors_per_piece = field[0, :, mask_canvas == mask_number]
                if vectors_per_piece.shape[1] == 0:
                    continue # all is ocluded
                average_translation = vectors_per_piece.mean(dim=1)
                avg_x, avg_y, _ = average_translation
                pieces[mask_number -1][0] -= int(torch.round(avg_x).item())
                pieces[mask_number - 1][1] -= int(torch.round(avg_y).item())


        # get the initial location of all pieces
        pieces = [list(p.draw_piece()) for p in self.pieces]
        for idx in range(len(pieces)):
            x, y, img, mask = pieces[idx]
            pieces[idx][0] = x + (self.pad[0] // 2)
            pieces[idx][1] = y + (self.pad[0] // 2)

        def single_step(step_index:int ):
            img, mask = draw_pieces(pieces)
            img = image_transforms((img / 255.0).astype(np.float32))
            img = torch.unsqueeze(img, 0)
            img = img.to(device=device)
            out = model(img, torch.tensor([step])) * 255
            move_pieces_according_to_field(out, mask, pieces)

        for step in range(initial_step, 0, -1):
            single_step(step)

        if resolve_step != 0:
            for step in range(resolve_step, 0, -1):
                single_step(step)

        final_image, _ = draw_pieces(pieces)
        return final_image





