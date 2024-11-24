from enum import Enum


class MaliciousType(Enum):
    NONE = "none"
    RANDOM_UPDATES = "random_updates"
    SCALED_UPDATES = "scaled_updates"
    CONSTANT_UPDATES = "constant_updates"
