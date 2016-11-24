from mnistComposer import get_random_digit
import numpy as np


def create_pairs():
    """
    simply creates pairs of digits.
    if its the same digit in both images, label == 0
    if its a different digit in both images, label == 1
    :return:
    """
    y = np.random.choice(2, size=1)[0]

    if y == 0:
        which_dig = np.argmax(np.random.multinomial(1, [0.1] * 10))
        Zleft = get_random_digit(which_dig)
        Zright = get_random_digit(which_dig)
    else:
        which_dig_left = np.argmax(np.random.multinomial(1, [0.1] * 10))
        which_dig_right = which_dig_left - 1 if which_dig_left != 0 else 9  # the second one is just on smaller than the first digit + bondary
        Zleft = get_random_digit(which_dig_left)
        Zright = get_random_digit(which_dig_right)
    X = np.stack([Zleft, Zright])
    return X,y


def plot_pair(X):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(X[0])
    plt.subplot(1, 2, 2)
    plt.imshow(X[1])