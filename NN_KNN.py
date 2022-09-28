import numpy as np
import tensorflow
import cv2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
dict_classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
                8: "ship", 9: "truck"}


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def most_frequent(list):
    list = [x[0] for x in list]
    return [max(set(list), key=list.count)]


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 1 - Step 3 - Compute the difference between the current image (img) taken from test dataset
#  with all the images from the train dataset. Return the label for the training image for which is obtained
#  the lowest score
def predictLabelNN(x_train_flatten, y_train, img):
    predictedLabel = -1
    score = 0
    scoreMin = 100000000

    # TODO - Application 1 - Step 3a - for each image in the training list
    for idx, imgT in enumerate(x_train_flatten):

        # TODO - Application 1 - Step 3b - compute the absolute difference between img and imgT
        difference = np.abs(img - imgT)

        # TODO - Application 1 - Step 3c - add all pixels differences to a single number (score)
        score = np.sum(difference)

        # TODO - Application 1 - Step 3d - retain the label where the minimum score is obtained
        if score < scoreMin:
            scoreMin = score
            predictedLabel = y_train[idx][0]

    return predictedLabel


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 2 - Step 1 - Create a function (predictLabelKNN) that predict the label for a test image based
#  on the dominant class obtained when comparing the current image with the k-NearestNeighbor images from train
def predictLabelKNN(x_train_flatten, y_train, img):
    global predictions
    predictions = []
    score = 0
    predictedLabel = -1
    #predictions = [score, y_train[idx]]
    # TODO - Application 2 - Step 1a - for each image in the training list
    for idx, imgT in enumerate(x_train_flatten):
        # TODO - Application 2 - Step 1b - compute the absolute difference between img and imgT
        difference = np.abs(img - imgT)

        # TODO - Application 2 - Step 1c - add all pixels differences to a single number (score)
        score = np.sum(difference)

        # TODO - Application 2 - Step 1d - store the score and the label associated to imgT into the predictions list
        #  as a pair (score, label)
        predictions.append((score, y_train[idx][0]))

    # TODO - Application 2 - Step 1e - Sort all elements in the predictions list in ascending order based on scores
    predictions = sorted(predictions, key=lambda x: x[0])

    # TODO - Application 2 - Step 1f - retain only the top k=10 predictions
    #res = {key: val for key, val in x_train(10) if val in x_train_flatten}
    top10predictions = predictions[:50]

    # TODO - Application 2 - Step 1g - extract in a separate vector only the labels for the top k predictions
    predLabels = (lambda x: x[1])(top10predictions)

    # TODO - Application 2 - Step 1h - Determine the dominant class from the predicted labels
    def most_frequent(list):
        list=[x[1] for x in list]
        return [max(set(list), key = list.count)]

    predictedLabel = most_frequent(top10predictions)
    return predictedLabel


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():
    # TODO - Application 1 - Step 1 - Load the CIFAR-10 dataset
    # input data and labels
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()

    # TODO - Exercise 1 - Determine the size of the four vectors x_train, y_train, x_test, y_test
    size1 = len(x_train)
    size2 = len(y_train)
    size3 = len(x_test)
    size4 = len(y_test)

    # TODO - Exercise 2 - Visualize the first 10 images from the testing dataset with the associated labels
    x_test_flatten = x_test.reshape(x_test.shape[0], 32 * 32 * 3)
    for idx, img in enumerate(x_test_flatten[0:10]):
        print("First images ".format(idx))

    # TODO - Application 1 - Step 2 - Reshape the training and testing dataset from 32x32x3 to a vector
    x_train_flatten = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_test_flatten = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

    numberOfCorrectPredictedImages = 0

    # TODO - Application 1 - Step 3 - Predict the labels for the first 200 images existent in the test dataset
    for idx, img in enumerate(x_test_flatten[0:200]):

        print("Make a prediction for image {}".format(idx))

        # TODO - Application 1 - Step 3 - Call the predictLabelNN function
        predictedLabel = predictLabelKNN(x_train_flatten, y_train, img)

        # TODO - Step 4 - Compare
        if predictedLabel == y_test[idx]:
            numberOfCorrectPredictedImages = numberOfCorrectPredictedImages + 1

    # TODO - Application 1 - Step 5 - Compute the accuracy
    accuracy = 100 * (numberOfCorrectPredictedImages / 200)
    print("System accuracy = {}".format(accuracy))

    return


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
