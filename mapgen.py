""" generate random terrain-like 2D image

    (c) Volker Poplawski 2018
"""
from PIL import Image, ImageFilter
import random


IMPASSABLE = 0, 0, 0
PASSABLE = 220, 220, 220


def generate_map(size, kernelsize, numiterations):
    im = Image.new('RGB', (size, size), color=IMPASSABLE)

    # init with random data
    for x in range(0, im.width):
        for y in range(0, im.height):
            im.putpixel((x, y), random.choice([IMPASSABLE, PASSABLE]))

    # apply filter multiple times
    for i in range(numiterations):
        im = im.filter(ImageFilter.RankFilter(kernelsize, kernelsize**2 // 2))

    return im




# from PIL import Image, ImageFilter
# import random
# import numpy as np
# IMPASSABLE = 0, 0, 0
# PASSABLE = 220, 220, 220
# def generate_map(size, kernelsize, numiterations):
#     maze = np.array([
#         [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
#         [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
#         [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
#         [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#         [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
#         [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
#         [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#         [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
#     ])
#     im = Image.fromarray(np.uint8(maze * 220), 'L')
#     newsize = (512, 512)
#     im = im.resize(newsize, resample=Image.BOX)
#     im = im.convert('RGB')
#     return im