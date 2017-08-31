# USAGE
# python mnist_dbn.py

# import the necessary packages
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from DataAndDescription.utils import dataset
from sklearn import datasets
from nolearn.dbn import DBN
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_error(epochs, errors_fine_tune):
    plt.plot(epochs, errors_fine_tune, '-', linewidth=2.0, label='error')
    plt.xlabel('Epochs (Number of times data is shown to the Network)')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Decline in Error during training')
    plt.xlim(np.min(epochs) - 0.003, np.max(epochs) + 0.003)
    plt.ylim(np.min(errors_fine_tune) - 0.003, np.max(errors_fine_tune) + 0.003)
    plt.show()

# grab the MNIST dataset (if this is your first time running this script, the
# download may take a minute -- the 55mb MNIST dataset will be downloaded)
print("[INFO] downloading MNIST...")
data, target = dataset.load_digits('data/digits.csv')
flattened_data = []

# normalize data and then construct the training and testing
# splits
for row in data:
    # flatten 28 x 28 image and normalize intensities
    flattened_row = row.flatten()/255.0
    flattened_data.append(np.array(flattened_row))

flattened_data = np.array(flattened_data)
(trainData, testData, trainLabels, testLabels) = train_test_split(
    flattened_data, target.astype("int"), test_size=0.33)


# train the Deep Belief Network with 784 input units (i.e., the flattened,
# 28 x 28 grayscale images), 300 hidden units, and 10 output units (one for
# each of the possible output classifications)
dbn = DBN(
    [trainData.shape[1], 300, 10],
    learn_rates=0.1,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1)

# train the model
dbn.fit(trainData, trainLabels)

# dump the model to file
joblib.dump(dbn, "models/dbn_model.pkl")