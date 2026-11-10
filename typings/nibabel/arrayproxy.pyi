"""Public scaled slab access provided by NiBabel 5.4 ArrayProxy."""

import numpy as np
from numpy.typing import NDArray

class ArrayProxy:
    def __getitem__(self, slicer: tuple[slice, slice, slice]) -> NDArray[np.generic]: ...
