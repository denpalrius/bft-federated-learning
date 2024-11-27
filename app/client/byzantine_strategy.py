from enum import Enum


class ByzantineStrategy(Enum):
    SIGN_FLIP = "sign_flip"
    GAUSSIAN_NOISE = "gaussian_noise"
    CONSTANT_BIAS = "constant_bias"
    ZERO_UPDATE = "zero_update"
