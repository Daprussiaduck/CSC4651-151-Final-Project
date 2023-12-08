import tensorflow as tf
import numpy as np
import cv2 as cv
import time

if __name__ == '__main__':
    # Print tensorflow version to ensure it is installed
    print("Tensorflow version:", tf.__version__)
    # load model
    print('Loading Model')
    model = tf.keras.models.load_model('./Final.keras')
    # open camera
    cap = cv.VideoCapture(0)
    windowName = 'Webcam'
    # while we have data
    frameNumber = 0
    batchSize = 32 # Batch size of the Dataset
    scalingFactor = 4 # scaling factor of the full image
    imgHeight = int(1920/scalingFactor) # desired image height
    imgWidth = int(1080/scalingFactor) # desired image width
    resizefn = tf.keras.layers.Resizing(imgHeight, imgWidth)
    preprocessinput = tf.keras.applications.resnet_v2.preprocess_input
    si = True
    while True:
        if not cap.isOpened():
            print('Unable to load camera.')
        # read frames
        ret, frame = cap.read()
        cv.imshow(windowName, frame)
        print('Received frame number', frameNumber, 'in the video buffer')
        frameNumber = frameNumber + 1
        #  predict
        if si:
            startTime = time.time()
            #image = resizefn(frame)
            #result = model.predict(image)
            result = model.predict(np.expand_dims(cv.resize(frame,  (270, 480), interpolation = cv.INTER_LINEAR), axis = 0))
            endTime = time.time() - startTime
            print(result, "took: ", endTime)
        si = not si
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
