import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

def apply_filter(image, kernel):
    padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    padded[1:-1, 1:-1] = image
    result = np.zeros_like(image)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            result[x, y] = (kernel * padded[x:x+3, y:y+3]).sum()
    return result

def main():
    kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

    img_path = 'lenna.png'
    img = plt.imread(img_path)

    num_chunks = 8
    chunk_height = img.shape[0] // num_chunks

    chunks = []
    for i in range(num_chunks):
        chunk = img[i * chunk_height:(i + 1) * chunk_height, :]
        chunks.append(chunk)

    start = time.time()

    with ProcessPoolExecutor(max_workers=num_chunks) as e:
        results = list(e.map(apply_filter, chunks, [kernel] * num_chunks))

    result = np.vstack(results)

    end = time.time()

    print(f'czas: {end - start}s')

    plt.imshow(result, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
