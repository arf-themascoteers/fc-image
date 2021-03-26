from zipfile import ZipFile
import os
import urllib
import urllib.request

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'data'
CHECK_FILE = "data/train"

if not os.path.exists(CHECK_FILE):
    urllib.request.urlretrieve(URL, FILE)
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)

labels = os.listdir( 'data/train' )
