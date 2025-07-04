import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class RandomShapeOverlay(nn.Module):
    """
    Додає до зображень випадкову білу фігуру:
    - 'circle'   : заповнений круг
    - 'square'   : заповнений квадрат‑патч
    - 'frame'    : рамка по периметру
    - 'gaussian' : біла ґаусова пляма (additive)

    Параметри:
        p               – імовірність застосувати трансформацію до всього батча
        shape_prob       – список/tuple ймовірностей для кожної фігури (у тому ж порядку, що shapes)
        radius_range     – (min, max) відносний радіус кола   (частка від min(H, W))
        square_frac_rng  – (min, max) частка сторони квадрата від min(H, W)
        frame_thickness  – відносна товщина рамки (частка від min(H, W))
        sigma_range      – (min, max) σ для ґаусової плями, частка від min(H, W)
        white_value      – значення, яким малюємо (1.0 для нормованих зображень)
        inplace          – True: змінює вхідний тензор; False: повертає копію
    """
    def __init__(
        self,
        p: float = 0.5,
        shapes=("circle", "square", "frame", "gaussian"),
        shape_prob=None,
        radius_range=(0.05, 0.25),
        square_frac_rng=(0.1, 0.4),
        frame_thickness=0.05,
        sigma_range=(0.05, 0.15),
        white_value=1.0,
        inplace: bool = False,
    ):
        super().__init__()
        self.p = p
        self.shapes = shapes
        self.shape_prob = shape_prob
        self.radius_range = radius_range
        self.square_frac_rng = square_frac_rng
        self.frame_thickness = frame_thickness
        self.sigma_range = sigma_range
        self.white_value = white_value
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        if not self.inplace:
            x = x.clone()

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        size_min = min(H, W)

        # сітка координат для всіх зображень за раз (нормована до [0, 1])
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device, dtype=dtype),
            torch.linspace(0, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )

        for b in range(B):
            shape = random.choices(self.shapes, weights=self.shape_prob, k=1)[0]

            if shape == "circle":
                r = random.uniform(*self.radius_range) * size_min
                cy, cx = random.random(), random.random()
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r / size_min) ** 2

                x[b, :, mask] = self.white_value

            elif shape == "square":
                frac = random.uniform(*self.square_frac_rng)
                side = frac * size_min
                # верхній лівий кут
                y0 = random.uniform(0, 1 - side / H)
                x0 = random.uniform(0, 1 - side / W)

                mask = (yy >= y0) & (yy <= y0 + side / H) & \
                       (xx >= x0) & (xx <= x0 + side / W)

                x[b, :, mask] = self.white_value

            elif shape == "frame":
                t = self.frame_thickness * size_min
                # товщина у відношенні до висоти/ширини
                t_y, t_x = t / H, t / W
                mask = (yy <= t_y) | (yy >= 1 - t_y) | \
                       (xx <= t_x) | (xx >= 1 - t_x)

                x[b, :, mask] = self.white_value

            elif shape == "gaussian":
                sigma = random.uniform(*self.sigma_range)
                cy, cx = random.random(), random.random()
                gauss = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) /
                                  (2 * (sigma ** 2)))
                gauss = gauss.unsqueeze(0)  # 1 × H × W
                x[b] = torch.clamp(x[b] + gauss * self.white_value, 0, 1)

        return x
