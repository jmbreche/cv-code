import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Generate WxH binary array to mask region (for debugging purposes)
def gen_mask(w, h, x1, x2, y1, y2):
    return np.array([[1 if x1 <= col and x2 >= col and y1 <= row and y2 >= row else 0 for col in range(w)] for row in range(h)])


# Generate Nx2 array containing sample fixation points (for debugging purposes)
def gen_fixations(w, h, n):
    return np.array([(np.random.randint(0, w), np.random.randint(0, h)) for i in range(n)])


# Generate WxH frequency map to count number of fixations inside segmented regions
def gen_freq(mask, fixations):
    freq = np.zeros((mask.shape[0], mask.shape[1]))
    np.add.at(freq, (fixations[:, 1], fixations[:, 0]), 1)
    
    return freq * mask


def main():
    w = 1500 
    h = 2048
    n = 100000000

    mask = gen_mask(w, h, .5 * w, .75 * w, .25 * h, .9 * h)

    fixations = gen_fixations(w, h, n)

    freq = gen_freq(mask, fixations)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(mask)
    axs[1].imshow(freq, cmap="hot", interpolation="nearest")
    
    plt.show()


if __name__ == "__main__":
    main()
