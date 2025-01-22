import numpy as np
import matplotlib.pyplot as plt
import time

def apply_filter(image, kernel):
    image_padded = np.pad(image, 1, 'constant')
    result = np.zeros_like(image)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            result[x, y] = (kernel * image_padded[x:x+3, y:y+3]).sum()
    return result

def main():
    kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

    img_path = 'lenna.png'
    img = plt.imread(img_path)

    start = time.time()
    result = apply_filter(img, kernel)
    end = time.time()

    print(f'czas: {end - start}s')

    plt.imshow(result, cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    main()
