import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics
import numpy as np
import pathlib
import os



class estimator:
  _estimator_type = ''
  classes_=[]
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self.classes_ = classes
  def predict(self, X):
    y_prob= self.model.predict(X)
    y_pred = y_prob.argmax(axis=1)
    return y_pred

if __name__ == '__main__':
    # Print tensorflow version to ensure it is installed
    print("Tensorflow version:", tf.__version__)
    # Check data to see if training/validation data exists (docker mounted that folder correctly, or folder exists if running native), commit sodoku if not
    if not os.path.isdir('/HaGRID/train'):
        raise FileNotFoundError('Cannot find Training Data')
    trainDir = pathlib.Path('/HaGRID/train').with_suffix('')
    # trainImageCount = len(list(trainDir.glob('*/*.jpg')))
    if not os.path.isdir('/HaGRID/test'):
        raise FileNotFoundError('Cannot find Validation Data')
    validDir = pathlib.Path('/HaGRID/test').with_suffix('')
    # load the dataset
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
    # load the models
    validDir = pathlib.Path('/HaGRID/test').with_suffix('')
    modelFolder = '/HaGRID/bigGood/'
    accuracy = []
    loss = []
    f = open(modelFolder + "acc-loss.txt", 'w')
    for i in range(1, 81):
        model = tf.keras.models.load_model(modelFolder + "{:02d}".format(i) + '.keras')
        print("Model:", "{:02d}".format(i))
        metrics = model.evaluate(validateDS)
        # predictions = model.predict(validateDS)
        print(metrics)
        f.write(str(i) + ": " + str(metrics))
        loss.append(metrics[0])
        accuracy.append(metrics[1])
    plt.plot(loss, range(1, 81), color = 'red')
    plt.plot(accuracy, range(1, 81), color = 'yellow')
    plt.savefig(modelFolder + 'acc-loss.png')
    f.close()
    model = tf.keras.models.load_model(modelFolder + 'Final.keras')
    model.summary()
    # y = np.concatenate([y for x, y in validateDS], axis=0)
    images, labels = tuple(zip(*validateDS))
    yTrue = np.array(labels)
    classifier = estimator(model, classNames)
    sklearn.metrics.plot_confusion_matrix(estimator = model, X = validateDS, y_true = yTrue)
    plt.savefig(modelFolder + 'confusionMatrix.png')