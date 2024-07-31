import numpy as np
from PIL import Image, ImageDraw

def create_masks(img_size, square_sizes):
    masks = {}

    for size in square_sizes:
        size = size 
        all_masks = []

        for i in range(3):
            for j in range(3):
                for dx in [0, size // 2]:
                    for dy in [0, size // 2]:
                        mask = np.zeros((img_size, img_size), dtype=np.uint8)
                        for x in range(i * size, img_size, 3 * size):
                            for y in range(j * size, img_size, 3 * size):
                                x_off = x + dx
                                y_off = y + dy
                                if x_off + size <= img_size and y_off + size <= img_size:
                                    mask[x_off:x_off + size, y_off:y_off + size] = 1
                        all_masks.append(mask)

        masks[size] = all_masks

    return masks

def overlay_mask(image, mask, square, grid_on, correct_grid_on):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    if grid_on:
        for y in range(0, image.size[1], square[2]):
            draw.line([(0, y), (image.size[0], y)], fill=(128, 128, 128, 128))
        for x in range(0, image.size[0], square[2]):
            draw.line([(x, 0), (x, image.size[1])], fill=(128, 128, 128, 128))
        x, y, size = square
        draw.rectangle([y, x, y + size - 1, x + size - 1], fill=(255, 165, 0, 96))

    if correct_grid_on:
        mask_overlay = Image.fromarray((mask * 128).astype(np.uint8), mode='L').convert('RGBA')
        mask_overlay = mask_overlay.point(lambda p: p * 0.5)
        overlay = Image.alpha_composite(overlay, mask_overlay)

    combined = Image.alpha_composite(image, overlay)
    return combined
