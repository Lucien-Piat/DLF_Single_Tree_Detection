import numpy as np


def detect_mask_edges(mask):
    # get the bounding coordinates of the mask
    mask_edges = np.zeros_like(mask)
    mask_edges += mask - mask[[-1, *list(range(0, len(mask)-1))], :]
    mask_edges += mask - mask[[*list(range(1, len(mask))), 0], :]
    mask_edges += mask - mask[:, [-1, *list(range(0, len(mask)-1))]]
    mask_edges += mask - mask[:, [*list(range(1, len(mask))), 0]]
    mask_edges = mask_edges > 0
    mask_edges = mask_edges.astype(np.uint8) * 255

    return mask_edges


def convert_to_yolo_list(mask_edges):
    yolo_list = []
    pass
