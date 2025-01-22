import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from z3_0 import apply_filter

def main():
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)

    img_path = 'lenna.png'
    img = plt.imread(img_path).astype(np.float32)

    start = time.time()
    result = apply_filter(img, kernel)
    end = time.time()

    print(f'czas: {end - start}s')
    plt.imshow(result, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
