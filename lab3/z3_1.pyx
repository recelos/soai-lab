from cython.parallel import prange
import numpy as np
import cython
cimport numpy as np
cimport cython

def apply_filter(np.float32_t[:, :] image, 
                 np.float32_t[:, :] kernel):
    cdef int x, y, i, j
    cdef int image_rows = image.shape[0]
    cdef int image_cols = image.shape[1]

    cdef np.float32_t[:,:] result = np.zeros_like(image)
    cdef np.float32_t[:,:] image_padded = np.pad(image, 1, 'constant')

    for x in prange(image_rows, nogil=True):
        for y in range(image_cols):
            result[x, y] = 0
            for i in range(3):
                for j in range(3):
                    result[x, y] = result[x, y] + kernel[i, j] * image_padded[x + i, y + j]

    return result
