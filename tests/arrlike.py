from typing import Tuple


class ImplementsArray:
    def __init__(self, shape: Tuple[int, ...]) -> None:
        assert all(i >= 0 and isinstance(i, int) for i in shape)
        self.shape = shape


def arr(*shape: int) -> ImplementsArray:
    return ImplementsArray(shape)
