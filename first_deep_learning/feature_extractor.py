import numpy as np


def convolution(image: np.ndarray, kernel: np.ndarray, stride: int, pad: int):
    h, w = image.shape
    t = np.zeros((h + 2 * pad,  w + 2 * pad), dtype=np.uint8)
    image = t[pad:pad+h, pad:pad+w] = image
    h, w = image.shape

    k_h, k_w = kernel.shape

    out_h = (h - k_h) // stride
    out_w = (w - k_w) // stride
    out = np.zeros((out_h, out_w), dtype=np.uint8)

    for y in range(out_h):
        for x in range(out_w):
            v = np.sum(image[y*stride:y*stride+k_h, x*stride:x*stride+k_w] * kernel)
            out[y][x] = v
    
    return out.astype(np.uint8)


def emphasize_bar(image: np.ndarray):
    kernel = np.array([
        [-1, -1, 0, 1, 1],
        [-1, -1, 0, 1, 1],
        [-1, -1, 0, 1, 1],
        [-1, -1, 0, 1, 1],
        [-1, -1, 0, 1, 1],
    ],
    dtype=np.float16
    )
    return convolution(image, kernel, stride=2, pad=2)


def average(image: np.ndarray):
    size = 9
    kernel = np.ones((size, size), dtype=np.float16) / (size * size)
    return convolution(image, kernel, stride=3, pad=3)


if __name__ == '__main__':
    import cv2
    from pathlib import Path
    import tqdm

    out = Path(__file__).parents[1] / "features"

    for image_path in tqdm.tqdm((Path(__file__).parents[1] / "dataset").glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = 2 * average(emphasize_bar(image))
        cv2.imwrite(str(out / image_path.name), image)
