import numpy as np
import image_viewer
from mnist import Mnist

mnist = Mnist()
train_x,train_y = mnist.get_train_data()
test_x,test_y = mnist.get_test_data()

train_x = (train_x.astype(np.float32) - 127.5 ) / 127.5
test_x = (test_x.astype(np.float32) - 127.5 ) / 127.5

train_x = train_x.reshape(train_x.shape[0], -1)
test_x = test_x.reshape(test_x.shape[0], -1)

keys = np.array(range(train_x.shape[0]))
np.random.shuffle(keys)
train_x = train_x[keys]

keys = np.array(range(test_x.shape[0]))
np.random.shuffle(keys)
test_x = test_x[keys]

image_viewer.view_image(train_x[0])

