# Adapted from https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
import numpy as np
from urllib import request
import gzip
import os

filename = [
    ["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
  base_url = "http://yann.lecun.com/exdb/mnist/"
  for name in filename:
    print("Downloading " + name[1] + "...")
    request.urlretrieve(base_url + name[1], name[1])
  print("Download complete.")

def save_mnist():
  mnist = {}
  for name in filename[:2]:
    with gzip.open(name[1], 'rb') as f:
      mnist[name[0]] = np.frombuffer(
          f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
  for name in filename[-2:]:
    with gzip.open(name[1], 'rb') as f:
      mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
  for name in filename:
    np.save(name[0] + '.npy', mnist[name[0]])
    os.remove(name[1])

def init():
  download_mnist()
  save_mnist()

if __name__ == '__main__':
  init()