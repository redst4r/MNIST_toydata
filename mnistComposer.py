from sklearn.datasets import fetch_mldata
import numpy as np

"""
to compose a large image out of smaller ones, aranging them somehow in space
"""


def load_mnist_dict():
    "dict of label -> np.array of digits"
    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data.astype(np.float32), mnist.target
    X=X.reshape(-1,28,28)

    return {i: X[y==i] for i in range(10)}


mnist_dict = load_mnist_dict()  # outside for performacne reasons (load the dict only once when sampling multiple digits)
def get_random_digit(which_dig):
    """
    return a random instance of the desired digit
    :param which_dig: one of [0,1,2,3,4,5,6,7,8,9]
    :return: 28x28 np.array
    """
    the_digs = mnist_dict[which_dig]
    ix = np.random.randint(0, len(the_digs))
    return the_digs[ix]


class ComposeImage(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width
        imsize = (height, width)
        self.image = np.zeros(imsize)
        self.mask = np.zeros(imsize)

    def _add(self, img_patch, h, w):
        "puts the image patch at the given position in the composed image. raises error if overlap"

        patch_H, patch_W = img_patch.shape
        assert h>=0 and w>=0 and h+patch_H < self.height and w+patch_W<self.width, "patch would not lie totally in the image"

        assert not self._has_overlap(h,w, patch_H, patch_W )
        ixH= np.arange(h,h+patch_H)
        ixW = np.arange(w, w + patch_W)

        ix_slice = np.ix_(ixH,ixW)  # careful here, just doing image[ixH,ixW] wont do what we wnat (that is  crossprod of ixH and ixW)
        self.image[ix_slice] = img_patch
        self.mask[ix_slice] = np.ones(img_patch.shape)

    def _has_overlap(self, h,w, patch_h, patch_w):
        "make sure that putting a patch of size (patch_h, patch_w) would not result in overlaps"
        ixH= np.arange(h,h+patch_h)
        ixW = np.arange(w, w + patch_w)

        ix_slice = np.ix_(ixH, ixW)
        return np.any(self.mask[ix_slice] == 1)

    def add_at_random_position(self, digit_img:np.ndarray):

        # look for a place to put it
        counter = 0
        while counter < 1000:  # try until we find find some place to put the thing; stop if we cannot find a place after 1000 trials
            h = np.random.randint(0,self.height-digit_img.shape[0])
            w = np.random.randint(0, self.width-digit_img.shape[1])
            if not self._has_overlap(h,w, digit_img.shape[0], digit_img.shape[1]):
                self._add(digit_img, h, w)
                break
            counter+=1
        return h, w


if __name__ == '__main__':

    mnist_dict = load_mnist_dict()
    C = ComposeImage(height=500, width=500)

    C.add_at_random_position(get_random_digit(which_dig=7))
    C.add_at_random_position(get_random_digit(which_dig=7))
    C.add_at_random_position(get_random_digit(which_dig=3))


    # plt.imshow(C.image)



