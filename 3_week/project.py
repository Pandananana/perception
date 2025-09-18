import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def sum_absolute_diff(template, test):
    # Convert to float to avoid uint8 overflow issues
    template_float = template.astype(np.float32)
    test_float = test.astype(np.float32)

    # Calculate sum of absolute difference
    abs_diff = np.abs(template_float - test_float)
    sad = np.sum(abs_diff)

    return int(sad)


@jit(nopython=True)
def match_line(template, line):
    matches = []
    for i in range(line.shape[1] - block_size + 1):
        test = line.copy()[0:block_size, i : i + block_size]
        sad = sum_absolute_diff(template, test)
        matches.append((sad, i))

    matches.sort()
    return matches[0][0]


@jit(nopython=True)
def create_disparity(left, right, block_size):
    disparity_map = np.zeros_like(left)
    for line in range(left.shape[0] - block_size + 1):
        print("Line: ", line)
        for pixel in range(left.shape[1] - block_size + 1):
            left_block = left.copy()[
                line : line + block_size, pixel : pixel + block_size
            ]
            right_line = right.copy()[line : line + block_size, 0 : left.shape[0]]
            pixel_match = match_line(left_block, right_line)
            pixel_disparity = np.abs(pixel - pixel_match)
            disparity_map[line, pixel] = pixel_disparity

    return disparity_map


if __name__ == "__main__":
    left = cv2.imread("tsukuba_left.png")
    right = cv2.imread("tsukuba_right.png")

    block_size = 5

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    map = create_disparity(left, right, block_size)

    plt.imshow(map, cmap="grey")
    plt.show()
