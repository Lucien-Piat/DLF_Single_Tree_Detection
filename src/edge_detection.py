import numpy as np


def detect_mask_edges(mask):
    # get the bounding coordinates of the mask
    mask_edges = np.zeros_like(mask)
    mask_edges += mask - mask[[-1, *list(range(0, len(mask)-1))], :]
    mask_edges += mask - mask[[*list(range(1, len(mask))), 0], :]
    mask_edges += mask - mask[:, [-1, *list(range(0, len(mask)-1))]]
    mask_edges += mask - mask[:, [*list(range(1, len(mask))), 0]]
    mask_edges = mask_edges > 0
    mask_edges = mask_edges.astype(np.uint8)

    return mask_edges


def convert_to_yolo_item(mask_edges):
    x, y = np.where(mask_edges)
    xy = (np
          .stack([x, y], axis=1)
          .astype(str)
          .flatten()
          .tolist())

    xy = " ".join(xy)

    return f"0 {xy}"


def get_yolo_items(mask):
    yolo_items = []

    for item_value in np.unique(mask):
        if item_value == 0:
            continue
        item_mask = (mask == item_value).astype(np.uint8)
        mask_edges = detect_mask_edges(item_mask)
        yolo_items.append(convert_to_yolo_item(mask_edges))

    return "\n".join(yolo_items)
