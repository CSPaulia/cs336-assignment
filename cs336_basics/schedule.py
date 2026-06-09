import math

def cosine_lr_schedule(
        t: int,
        lr_max: float,
        lr_min: float,
        t_warmup: int,
        t_cosine: int
    ) -> float:

    if t < t_warmup:
        return lr_max * t / t_warmup
    elif t < t_cosine:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - t_warmup) / (t_cosine - t_warmup)))
    else:
        return lr_min