import progressbar
from mnistComposer import ComposeImage, load_mnist_dict
import numpy as np

"""
in the spirit of this Kraus et al. Bioinformatics paper on Multi instance learning
"""

mnist_dict = load_mnist_dict()  # TODO dangerous, global var


def create_some_MIL_dataset(n_samples=10, imsize=(300,300), n_digits=100):
    """
    creates a dataset with images mixed of digits of zero and 5 and 7
    theres two classes: the one contains 90% zeros, 10% fives
                        the other contains 90% zeros, 10% sevens
    :return:
    """
    X, y = [],[]

    bar = progressbar.ProgressBar()
    for _ in bar(range(n_samples)):
        if np.random.rand() < 0.5:
            the_class = 0
            the_digit = 5
        else:
            the_class = 1
            the_digit = 7

        C = ComposeImage(height=imsize[0], width=imsize[1])

        # create 90% zeros
        digits0 = int(n_digits*0.9)
        digits_other = n_digits - digits0

        add_digits(C, which_dig=the_digit, N=digits_other)
        add_digits(C, which_dig=0, N=digits0)

        X.append(C.image.astype('float32'))
        y.append(the_class)

    return np.stack(X), np.stack(y)


def more_complicated_dataset(n_samples=10, imsize=(300,300), n_digits=100, shrink_factor=1):
    """
    3 classes:
    class 0:  only contains zeros
    class 1: contains zeros and ones
    class 2: contains zeros, ones and twos
    :param n_samples:
    :param imsize:
    :param n_digits:
    :return:
    """
    X, y = [], []

    class_digit_composition = {0: [0.9,0.05,0.05],        # only zeros, but some noise
                               1: [0.7, 0.25, 0.05],  # mostly 0 some 1
                               2: [0.1, 0.1, 0.8]}  # mostly 0 zome 1,2

    bar = progressbar.ProgressBar()
    for _ in bar(range(n_samples)):
        r = np.random.rand()
        if r < 0.33:
            the_class = 0
        elif 0.33 <= r < 0.66:
            the_class = 1
        else:
            the_class = 2

        C = ComposeImage(height=imsize[0], width=imsize[1])

        the_digits = np.random.choice([0,1,2], size=n_digits, replace=True, p=class_digit_composition[the_class])
        for i in range(len(the_digits)):
            add_digits(C, which_dig=the_digits[i], N=1, downsize_factor=shrink_factor)

        X.append(C.image.astype('float32'))
        y.append(the_class)

    return np.stack(X), np.stack(y)


def max_dataset(n_samples=10, imsize=(300,300), n_digits=100, shrink_factor=1, n_cores=1):
    """
    four digits: 0,1, 2, 3,
    if only zeros present: class0 (rest< 5%)
    if at least 25% 1's present (but no hiher) -> class1
    if at least 25% 2's present, class 2
    ...
    :param n_samples:
    :param imsize:
    :param n_digits:
    :return:
    """
    class_digit_composition = {0: [0.85,0.05,0.05, 0.05],        # only zeros, but some noise
                               1: [0.35, 0.55, 0.05, 0.05],  # mostly 0 some 1
                               2: [0.20, 0.3, 0.45, 0.05],  # mostly 0 zome 1,2
                               3: [0.15, 0.2, 0.25, 0.40]}  # mostly 0 zome 1,2

    bar = progressbar.ProgressBar()

    if n_cores == 1:
        X, y, digits_features = [],[],[]  # digits_features: count if the different digits in the picture, given that good classifiaction should be possible
        for _ in bar(range(n_samples)):

            img, label, f = _max_data_single_sample(imsize, n_digits, class_digit_composition, shrink_factor)
            X.append(img)
            y.append(label)
            digits_features.append(f)
    else:
        import multiprocessing
        with multiprocessing.Pool(processes=n_cores) as pool:

            results = [pool.apply_async(_max_data_single_sample, args=(imsize, n_digits, class_digit_composition, shrink_factor))
                       for _ in range(n_samples)]
            # the async call doesnt yield the returns of the function immediatly
            output = [p.get() for p in results]
        X,y,digits_features = zip(*output)

    return np.stack(X), np.stack(y), np.stack(digits_features)


def _max_data_single_sample(imsize, n_digits, class_digit_composition, shrink_factor):
    "creates a single sample for the max_data dataset. see max_dataset() documentation"
    the_class = np.random.choice(4)
    C = ComposeImage(height=imsize[0], width=imsize[1])

    the_digits = np.random.choice([0, 1, 2, 3], size=n_digits, replace=True, p=class_digit_composition[the_class])
    for i in range(len(the_digits)):
        add_digits(C, which_dig=the_digits[i], N=1, downsize_factor=shrink_factor)

    X = C.image.astype('float32')
    y = the_class

    features = np.array([np.sum(_ == the_digits) for _ in [0, 1, 2, 3]])

    return X,y,features


def add_digits(composed_image, which_dig, N, downsize_factor=1):

    "adds N instances of the requested class to the compose image"
    "downsize_factors: 0.5 -> dont add the digit in full resolution, but shrinked to 0.4"
    for i in range(N):
        the_digs = mnist_dict[which_dig]
        ix = np.random.randint(0, len(the_digs))
        Z = the_digs[ix]
        Z = downsize_img(Z, factor=downsize_factor)
        composed_image.add_at_random_position(Z)

# downsize it
def downsize_img(I, factor):
    from scipy.interpolate import griddata
    H,W = I.shape
    grid_x, grid_y = np.mgrid[0:H, 0:W]
    i_flat = I.flatten()
    x_flat = grid_x.flatten()
    y_flat = grid_y.flatten()
    points = np.vstack([x_flat,y_flat]).T
    new_x, new_y = np.mgrid[0:H:factor, 0:W:factor]
    grid_z2 = griddata(points, i_flat, (new_x, new_y), method='cubic')
    return grid_z2
