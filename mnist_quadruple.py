from mnistComposer import ComposeImage, get_random_digit
import numpy as np


def add_symmetric_digits(comp1:ComposeImage, comp2:ComposeImage):
    """
    given two ComposeImages, add instances of the same digit to each image at the same location
    :param comp1: ComposeImage instance
    :param comp2: ComposeImage instance
    :return: None, alters the inputs
    """
    which_dig = np.argmax(np.random.multinomial(1, [0.1] * 10))
    Zleft = get_random_digit(which_dig)
    Zright = get_random_digit(which_dig)

    h,w = comp1.add_at_random_position(Zleft)  # add to the first image
    comp2._add(Zright, h, w)  # add in the second image at the same position


def add_asymmetric_digits(comp1:ComposeImage, comp2:ComposeImage, mode):
    """
    given two ComposeImages, add instances of two different digits to each image at the same location
    :param comp1: ComposeImage instance
    :param comp2: ComposeImage instance
    :param mode: if 'missing' just add the digit to one image
                 if 'difference' add a different digit to the other image (e.g. 0 to comp1 and 1 to comp2)

    :return: None, alters the inputs
    """
    assert mode in ['missing', 'different']
    "just add the digit into the first image, but not the second"
    which_dig1 = np.argmax(np.random.multinomial(1, [0.1] * 10))
    Zleft = get_random_digit(which_dig1)

    h,w = comp1.add_at_random_position(Zleft)  # add to the first image

    if mode == 'missing':
        pass
    elif mode == 'different':
        which_dig2 = which_dig1 - 1 if which_dig1 != 0 else 9  # the second one is just on smaller than the first digit + bondary
        Zright = get_random_digit(which_dig2)
        comp2._add(Zright, h, w) # add in the second image at the same position
    else:
        raise ValueError('unkown mode: %s' % mode)


def create_sample(n_digits, H, W, mode):
    """
    creates four images with labels 0,1,2, containing a few mnist digits

    0: both images are almost symmetric, contain same numbers at same location
    1: a single digits is missing right
    2: a single digit is missing left
    :param n_digits:
    :param mode: missing: just leave out a digit in one image
                 different: put in a differnet image in one of the images (contains the digit decreased by one)
                            THIS IS REALLY HARD TO DETECT I GUESS (0: contan the same; easy --
                                                                   1: left image has one digit decreased)
                                                                   2: right image has one digit decreased)
    :return:
    """
    left1 = ComposeImage(height=H, width=W)
    left2 = ComposeImage(height=H, width=W)

    right1 = ComposeImage(height=H, width=W)
    right2 = ComposeImage(height=H, width=W)

    y = np.random.choice(3, size=1)[0]

    if y == 0:  # always add symmetrically
        for i in range(n_digits):
            add_symmetric_digits(left1, right1)
            add_symmetric_digits(left2, right2)
    elif y == 1:  # missing thing right
        for i in range(n_digits-1):
            add_symmetric_digits(left1, right1)
            add_symmetric_digits(left2, right2)

        add_asymmetric_digits(left1,right1, mode)
        add_asymmetric_digits(left2, right2, mode)

    elif y == 2:  # missing thing left
        for i in range(n_digits-1):
            add_symmetric_digits(left1, right1)
            add_symmetric_digits(left2, right2)
            # ADD ONE ASSYMETRY, missing digit left
        add_asymmetric_digits(right1, left1, mode)
        add_asymmetric_digits(right2, left2, mode)

    I =  np.stack([x.image.astype('float32') for x in [left1, right1, left2, right2]])
    return I, y


def plot4img(I):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(I[0])

    plt.subplot(2, 2, 2)
    plt.imshow(I[1])

    plt.subplot(2, 2, 3)
    plt.imshow(I[2])

    plt.subplot(2, 2, 4)
    plt.imshow(I[3])
