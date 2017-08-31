# USAGE
# python eigenfaces.py --dataset caltech_faces

# import the necessary packages
from __future__ import print_function

import argparse

import imutils
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import cv2
from FaceRecognition import ResultsMontage
from FaceRecognition.face_recognition.datasets import load_caltech_faces

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to CALTECH Faces dataset")
ap.add_argument("-n", "--num-components", type=int, default=150, help="# of principal components")
ap.add_argument("-s", "--sample-size", type=int, default=20, help="# of example samples")
ap.add_argument("-v", "--visualize", type=int, default=-1,
			    help="whether or not PCA components should be visualized")
args = vars(ap.parse_args())

# load the CALTECH faces dataset
print("[INFO] loading CALTECH Faces dataset...")
(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21, flatten=True,
	test_size=0.25)

# compute the PCA (eigenfaces) representation of the data, then project the training data
# onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")
pca = RandomizedPCA(n_components=args["num_components"], whiten=True)
trainData = pca.fit_transform(training.data)

# plot variance in all the components
plt.plot(xrange(0, trainData.shape[1]), pca.explained_variance_, 'bo')
plt.xlabel('Components')
plt.ylabel('Eigenvalues')
plt.title('Principal Components and Associated Eigenvalues')
plt.show()

# check to see if the PCA components should be visualized
if args["visualize"] > 0:
	# initialize the montage for the components
	montage = ResultsMontage((62, 47), 5, 5)

	# loop over the first 16 individual components
	for (i, component) in enumerate(pca.components_[:5]):
		# reshape the component to a 2D matrix, then convert the data type to an unsigned
		# 8-bit integer so it can be displayed with OpenCV
		component = component.reshape((62, 47))
		component = exposure.rescale_intensity(component, out_range=(0, 255)).astype("uint8")
		component = np.dstack([component] * 3)
		montage.addResult(component)

	# show the mean and principal component visualizations
	# show the mean image
	mean = pca.mean_.reshape((62, 47))
	mean = exposure.rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
	cv2.imshow("Mean", mean)
	cv2.imshow("Components", montage.montage)
	cv2.waitKey(0)

# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=84)
model.fit(trainData, training.target)

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testing.data))
print(classification_report(testing.target, predictions))

# Loop over the the desired number of samples
classifiedMontage = ResultsMontage((62, 47), 5, args["sample_size"])
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
	#Predict image using the trained model
	prediction = model.predict(pca.transform(testing.data[i]))
	number = int(filter(str.isdigit, str(prediction)))#Extract number

	#Reshape the image for visualization
	image = testing.data[i].reshape((62, 47))
	image = exposure.rescale_intensity(image, out_range=(0, 255)).astype("uint8")
	image = np.dstack([image] * 3)

    #Draw the label and add to montage
	cv2.putText(image, str(number), (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))#draw text on image
	classifiedMontage.addResult(image)

cv2.imshow("Prediction montage", classifiedMontage.montage)
cv2.waitKey(0)