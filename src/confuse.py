import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics
import numpy as np
import pathlib
import os
import re

class estimator:
    _estimator_type = ''
    classes_=[]
    def __init__(self, model, classes):
        self.model = model
        self._estimator_type = 'classifier'
        self.classes_ = classes
    def predict(self, X):
        y_prob= self.model.predict(X)
        # y_pred = accuracy_score(y_prob, np.argmax(X, axis=1))
        return y_pred


def accLossGraph():
    acc = []
    loss = []
    with open('./acc-loss.txt') as f:
        for i in f.readlines():
            loss.append(float(i[(re.search('\[[0-9]+.[0-9]+,', i)).start() + 1 : (re.search('\[[0-9]+.[0-9]+,', i)).end() - 1]))
            # print(loss)
            acc.append(float(i[(re.search('[0-9]+.[0-9]+\]', i)).start() : (re.search('[0-9]+.[0-9]+]', i)).end() - 1]))
            # print(acc)
    plt.plot(range(1, 81), loss, color = 'red', label = 'Loss')
    plt.plot(range(1, 81), acc, color = 'orange', label = 'Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('./acc-loss.png')
    plt.show()

def confusionMatrix():
    print('Starting confusion matrix')
    validDir = pathlib.Path('/PATH_TO_DATASET/HaGRID/test').with_suffix('')
    batchSize = 32 # Batch size of the Dataset
    scalingFactor = 4 # scaling factor of the full image
    imgHeight = int(1920/scalingFactor) # desired image height
    imgWidth = int(1080/scalingFactor) # desired image width
    resizefn = tf.keras.layers.Resizing(imgHeight, imgWidth)
    preprocessinput = tf.keras.applications.resnet_v2.preprocess_input
    validateDS = tf.keras.utils.image_dataset_from_directory(validDir, image_size = (imgHeight, imgWidth), batch_size = batchSize, label_mode = 'categorical')
    validateDS.map(lambda x, y: (resizefn(x), y))
    validateDS.map(lambda x, y: (preprocessinput(x), y))
    classNames = validateDS.class_names
    print('Loading Model')
    model = tf.keras.models.load_model('./Final.keras')
    print('model loaded')
    y = np.concatenate([y for x, y in validateDS], axis = 0)
    # images, labels = tuple(zip(*validateDS))
    print('labels stripped')
    yTrue = y
    # print(y.shape)
    # yTrue = validateDS.labels
    print('labels ingested')
    print('predicting dataset')
    # classifier = estimator(model, classNames)
    print('plotting confusion matrix')
    # cM = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(classifier, validateDS, yTrue, display_labels = classNames)
    # cM.ax_.set_title("Confusion matrix for HaGRID Dataset")
    predictRes = model.predict(validateDS)
    print(predictRes)
    print(yTrue)
    confuzz = tf.math.confusion_matrix(np.argmax(yTrue, axis = 1), np.argmax(np.rint(predictRes), axis = 1))
    print("Confusion Matrix:", confuzz)
    with open('./confusionMatrix.txt', 'w') as f:
        f.write(str(confuzz))
    fig, ax = plt.subplots(figsize=(20,20))
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true = np.argmax(yTrue, axis = 1), y_pred = np.argmax(np.rint(predictRes), axis = 1), display_labels = classNames, ax = ax, xticks_rotation = 60)
    plt.savefig('./confusionMatrix.png')
    plt.show()
    # plt.show()

if __name__ == "__main__":
    # Print tensorflow version to ensure it is installed
    print("Tensorflow version:", tf.__version__)
    # accLossGraph()
    confusionMatrix()
