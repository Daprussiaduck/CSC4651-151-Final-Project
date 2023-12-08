# CSC4651-151-Final-Project
Final Project for the CSC4651-151 Class at MSOE Fall Semester 2023/2024

This is copied from my private repository for the class, but stripped down to be only the final project.<br/>
The release is the final trained network, the presentation and a video of the presentation (Unfortunatley, the theme of the presentation updated after recording but before PDF export, so slides are slightly different from the video)

Uses Tensorflow version 2.14, 'latest' has since been updated and loading network in a newer tensorflow version causes an error 

camera.py, confuse.py and results.py all require Final.keras in root of the repo
confuse.py, results.py, and all training files require the dataset, either mapped through docker or locally, can modify path accordingly


Code:

    src
        camera.py - Runs the program with webcam as the input (Requires OpenCV2 installed [both its pytho package and system install])
        confuse.py - Can print the Confusion Matrix and the Accuracy/Loss function (Requires Sci-py python package to be installed)
        train.py - Trains a new network (In Docker)
        
    not used, old code
        results.py - Attempt at confuse.py, but didn't work too well and didn't break Accuracy-Loss and Confusion into two separate callable (Requires Sci-py package to be installed)
        train2.py - Was used to take the output from train.py and add another 20 epochs under a v2 folder
        train3.py - Was used similar to train2.py, but added 40 epochs and put them under a v3 folder
        trainSmall.py - Same as train.py, but reduced image size to about 512x512
        trainSmall2.py - Same as train2.py, but for the trainSmall.py network output
