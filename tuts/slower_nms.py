import numpy as np
import cv2
import matplotlib.pyplot as plt


def slower_nms(boxes, overlapThresh):
    # if there is no boxes at all
    if len(boxes) == 0:
        return []

    # initialize the picked list
    pick = []

    # grab the coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # calculate the area of the bboxes and sort them by their bottom-right corner y-coordinates
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)  # default order is ascending

    while len(idxs) > 0:
        # grab the last element in the indexes, add it to the picked list and
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]


if __name__ == '__main__':
    a = np.arange(0, 10)
    print(np.maximum(5, a))

    b = np.random.random((10,))
    print(np.where(b < 0.5))
