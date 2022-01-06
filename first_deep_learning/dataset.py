import cv2
import tqdm
from typing import Callable
import numpy as np
import albumentations as alb
import random
from pathlib import Path


MARK_COLOR = 255

# create mahjong tile
# mark_function should receive and return np.ndarray (range: [0, 255], shape: (height, width), dtype: np.uint8))
def create_mahjong_tile(height: int, width: int, mark_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    tile = np.zeros((height, width), dtype=np.uint8)
    return mark_function(tile)

def draw_bamboo(tile: np.ndarray, height: int, width: int, x: int, y: int):
    tile[y:y+height, x] = MARK_COLOR
    tile[y, x-width//2:x+width//2+1] = MARK_COLOR
    tile[y+height//2, x-width//2:x+width//2+1] = MARK_COLOR
    tile[y+height, x-width//2:x+width//2+1] = MARK_COLOR


def two_bamboos(tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    margin = h // 10
    bamboo_height = (h - 3 * margin) // 2
    bamboo_width = w // 8
    draw_bamboo(tile, bamboo_height, bamboo_width, w//2, margin)
    draw_bamboo(tile, bamboo_height, bamboo_width, w//2, h-margin-bamboo_height)
    return tile

def three_bamboos(tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    margin = max(h // 10, 1)
    bamboo_height = (h - 3 * margin) // 2
    bamboo_width = max(w // 8, 2)
    draw_bamboo(tile, bamboo_height, bamboo_width, w//2, margin)
    draw_bamboo(tile, bamboo_height, bamboo_width, w//4, h-margin-bamboo_height)
    draw_bamboo(tile, bamboo_height, bamboo_width, w//4 * 3, h-margin-bamboo_height)
    return tile

def four_bamboos(tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    margin = h // 10
    bamboo_height = (h - 3 * margin) // 2
    bamboo_width = w // 8
    draw_bamboo(tile, bamboo_height, bamboo_width, w//4, margin)
    draw_bamboo(tile, bamboo_height, bamboo_width, w//4, h-margin-bamboo_height)
    draw_bamboo(tile, bamboo_height, bamboo_width, 3 * w//4, margin)
    draw_bamboo(tile, bamboo_height, bamboo_width, 3 * w//4, h-margin-bamboo_height)
    return tile

def randomize_tile(tile: np.ndarray) -> np.ndarray:
    compose = alb.Compose(
        [
            alb.Rotate(always_apply=True, limit=5),
        ]
    )
    return compose(image=tile)["image"]

def create_random_tiles(n_tiles: int, size: int, out_dir: Path):
    functions = [two_bamboos, three_bamboos, four_bamboos]
    names = ["2_bamboos", "3_bamboos", "4_bamboos"]
    height = size - 4
    width = height // 2
    for n in tqdm.tqdm(range(n_tiles * len(functions))):
        f_i = n // n_tiles
        tile = randomize_tile(create_mahjong_tile(height, width, functions[f_i]))
        image = np.zeros((size, size), dtype=np.uint8)
        image[(size-height)//2:(size-height)//2+height, (size-width)//2:(size-width)//2+width] = tile
        cv2.imwrite(str(out_dir / f"{names[f_i]}_{n:04}.png"), image)


if __name__ == "__main__":
    create_random_tiles(15, 40, Path(__file__).parents[1] / "dataset")