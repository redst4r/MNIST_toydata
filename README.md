# MNIST_toydata
some code to compose MNIST digits into more complex images/arangements

## mnistComposer.py
convenient class to compose a large image out of smaller ones

## mnist_pairs.py
just creates pairs of MNIST digits (as 2 channels) and a label what indicates wether or not the same digit appears in both channels

## mnist_quadruple.py
A more complicated dataset is created: 
- a single sample consists of 2x2 images in total
- a single image contains a couple of MNISt digits
- there's a total of 3 classes:
  - 0 if both pairs of images contain the same digits at the same location
  ```
  pair 1
  left             right
  ----------      ----------
  | 2      |      | 2      |
  |      6 |      |      6 |
  |   5    |      |   5    |
  ----------      ----------
  
  pair 2
  left             right
  ----------      ----------
  |     1  |      |     1  |
  |  3     |      |  3     |
  |   3    |      |   3    |
  ----------      ----------
  
  ```
  - 1 if a digit is missing/altered on the left
  
  ```
  pair 1
  left             right
  ----------      ----------
  | 1      |      | 2      |
  |      6 |      |      6 |
  |   5    |      |   5    |
  ----------      ----------
  
  pair 2
  left             right
  ----------      ----------
  |     1  |      |     1  |
  |  3     |      |  3     |
  |   4    |      |   3    |
  ----------      ----------
  
  ```
  - 2 if a digit is missing/altered on the right
  
  ## TODO 
  not clear how one can dstinguish 1 and 2, not possible to tell which one is the original and which on altered
 
