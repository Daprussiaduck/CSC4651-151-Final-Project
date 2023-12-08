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
    # ResNet152V2 ?
    baseModel = tf.keras.applications.ResNet152V2(weights = 'imagenet', input_shape = (imgHeight, imgWidth, 3), include_top = False, classes = len(classNames))
    baseModel.trainable = False
    # Start modifying base model for transfer learning - https://keras.io/guides/transfer_learning/
    inputs = tf.keras.Input(shape = (imgHeight, imgWidth, 3))
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = baseModel(x, training = False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2048, activation = tf.nn.relu)(x)
    x = tf.keras.layers.Dense(1024, activation = tf.nn.relu)(x)
    x = tf.keras.layers.Dense(512, activation = tf.nn.relu)(x)
    x = tf.keras.layers.Dense(256, activation = tf.nn.relu)(x)
    x = tf.keras.layers.Dense(128, activation = tf.nn.relu)(x)
    x = tf.keras.layers.Dense(64, activation = tf.nn.relu)(x)
    x = tf.keras.layers.Dense(32, activation = tf.nn.relu)(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(len(classNames), activation = tf.nn.softmax)(x)
    model = tf.keras.Model(inputs, outputs)
    # Compile the new model
    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy()])
    # Setup model checkpoints
    time = datetime.now().date() # current time
    checkpointDirName = '/HaGRID/' + str(time)
    os.makedirs(checkpointDirName, mode = 0o777)
    checkpointDir = os.path.dirname(checkpointDirName)
    print(checkpointDirName)
    checkpointFile = checkpointDirName + '/{epoch:02d}.keras'
    # The callback to save the model after every epoch
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointFile, save_best_only = False, save_weights_only = False, verbose = 1)
    #show the model summary
    model.summary()
    # Train new model
    model.fit(trainingDS, epochs = 80, validation_data = validateDS, callbacks = [checkpointCallback])
    # Save model after training
    model.save('/HaGRID/' + str(time) + "/Final.keras")
