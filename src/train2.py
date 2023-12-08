from datetime import datetime
import tensorflow as tf
import numpy as np
import pathlib
import os

if __name__ == '__main__':
    # Print tensorflow version to ensure it is installed
    print("Tensorflow version:", tf.__version__)
    # session = tf.session()
    # session.run(tf.local_variables_initializer())
    # session.run(tf.global_variables_initializer())
    # Check data to see if training/validation data exists (docker mounted that folder correctly, or folder exists if running native), commit sodoku if not
    if not os.path.isdir('/HaGRID/train'):
        raise FileNotFoundError('Cannot find Training Data')
    trainDir = pathlib.Path('/HaGRID/train').with_suffix('')
    # trainImageCount = len(list(trainDir.glob('*/*.jpg')))
    if not os.path.isdir('/HaGRID/test'):
        raise FileNotFoundError('Cannot find Validation Data')
    validDir = pathlib.Path('/HaGRID/test').with_suffix('')
    # valImageCount = len(list(valDir.glob('*/*.jpg')))
    # Parameters for the Image loader
    batchSize = 32 # Batch size of the Dataset
    scalingFactor = 4 # scaling factor of the full image
    imgHeight = int(1920/scalingFactor) # desired image height
    imgWidth = int(1080/scalingFactor) # desired image width
    resizefn = tf.keras.layers.Resizing(imgHeight, imgWidth)
    preprocessinput = tf.keras.applications.resnet_v2.preprocess_input
    # Load the image data set
    trainingDS = tf.keras.utils.image_dataset_from_directory(trainDir, image_size = (imgHeight, imgWidth), batch_size = batchSize, label_mode = 'categorical')
    trainingDS.map(lambda x, y: (resizefn(x), y))
    trainingDS.map(lambda x, y: (preprocessinput(x), y))
    # augment data here
    validateDS = tf.keras.utils.image_dataset_from_directory(validDir, image_size = (imgHeight, imgWidth), batch_size = batchSize, label_mode = 'categorical')
    validateDS.map(lambda x, y: (resizefn(x), y))
    validateDS.map(lambda x, y: (preprocessinput(x), y))
    # augment data here
    classNames = trainingDS.class_names
    print(classNames)
    # Setup caching on the data sets
    AUTOTUNE = tf.data.AUTOTUNE
    # trainingDS.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    trainingDS.cache().prefetch(buffer_size = AUTOTUNE)
    validateDS.cache().prefetch(buffer_size = AUTOTUNE)
    # Load the base model
    modelFolder = '/HaGRID/bigGood'
    model = tf.keras.models.load_model(modelFolder + '/done.keras')
    # Setup model checkpoints
    time = datetime.now().date() # current time
    checkpointDirName = modelFolder + '/v2'
    os.makedirs(checkpointDirName, mode = 0o777)
    checkpointDir = os.path.dirname(checkpointDirName)
    print(checkpointDirName)
    checkpointFile = checkpointDirName + '/{epoch:02d}_v2.keras'
    # The callback to save the model after every epoch
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointFile, save_best_only = False, save_weights_only = False, verbose = 1)
    #show the model summary
    model.summary()
    # Train new model
    model.fit(trainingDS, epochs = 20, validation_data = validateDS, callbacks = [checkpointCallback])
    # Save model after training
    model.save(modelFolder + "/done_v2.keras")
