from typing import Tuple
import cv2
from PIL import Image
import random
import networkx


def random_border_location(image_size, edge):
    x, y = 0,0
    if edge == 0 or edge == 2: # left or right border
        x = 0 if edge == 0 else image_size[0] - 1
        y = random.randint(0, image_size[1] - 1)

    else: # up or down border
        x = 0 if edge == 1 else image_size[1] - 1
        y = random.randint(0, image_size[0] - 1)


def random_cut(image_size):
    possible_borders = [0, 1, 2, 3]
    borders_to_cut = random.sample(possible_borders, k=2)
    borders_to_cut.sort()
    return random_border_location(image_size, borders_to_cut[0]), random_border_location(image_size, borders_to_cut[1])


def cut_to_line(cut):
    p1, p2 = cut
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def cut_intersection(cut1, cut2):
    L1, L2 = cut_to_line(cut1), cut_to_line(cut2)
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return True, x, y
    else:
        return False, 0, 0

def is_point_in_image(point, image_size):
    x,y = point
    return (0 <= x < image_size[0]) and (0 <= y < image_size[1])

def cuts_to_graph(cuts, image_size):
    cut_points = {}
    for i in range(len(cuts)):
        for j in range(i+1, len(cuts)):
            is_cut, x, y = cut_intersection(cuts[i], cuts[2])
            if not is_cut:
                continue

            pixel_cut = int(x), int(y)
            if pixel_cut in cut_points.keys():
                cut_points[pixel_cut].append()





def graph_to_pieces(graph):
    pass


def cut_image(original_image: str, number_of_cuts: int):
    """
    Cuts an image into peices
    :param original_image:
    :param number_of_cuts:
    :return:
    """

    image = Image.open(original_image)
    cuts = [random_cut(image.shape[:2]) for i in range(number_of_cuts)]
    cuts = list(set(cuts)) # filter out duplicates
    cut_graph = cuts_to_graph(cuts, image.shape)
