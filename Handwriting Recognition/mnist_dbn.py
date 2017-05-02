# USAGE
# python mnist_dbn.py

# import the necessary packages
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from DataAndDescription.utils import dataset
from sklearn import datasets
from nolearn.dbn import DBN
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
    [trainData.shape[1], 200, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1)

# train the model
dbn.fit(trainData, trainLabels)

losses_fine_tune = dbn.losses_fine_tune_
errors_fine_tune = dbn.errors_fine_tune_
epochs = dbn.epochs

epochs = np.arange(1, epochs + 1)

# plot error decrement through the epochs
plot_error(epochs, errors_fine_tune)	

# compute the predictions for the test data and show a classification report
print("[INFO] evaluating...")
predictions = dbn.predict(testData)
print(classification_report(testLabels, predictions))

# randomly select a few digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    prediction = dbn.predict(np.atleast_2d(testData[i]))

    # reshape the feature vector to be a 28 x 28 pixel image, then resize it
    # so we can visualize it better
    image = (testData[i] * 255).reshape((28, 28)).astype("uint8")
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

    # show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], testLabels[i]))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)