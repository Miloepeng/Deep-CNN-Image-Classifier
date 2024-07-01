# Deep-CNN-Image-Classifier
Binary Image Classifier using Tensorflow 
Tutorial by : https://youtu.be/jztwpsIzEGc?si=MkzW47aAFYFpsjSs

Using publicly available datasets, the Convulutional Neural Network was trained to identify happy or sad human faces. Evaluation of the model yield positive results.

The model was also trained to identify diabetic wounds amongst four different wound types: venous, diabetic, pressure, and surgical, inclusive of normal skin images without wounds. Despite positive evaluation results, the model had
trouble identifying diabetic wounds when images were selected and asked to be identified, often resulting in False Positive results. A similiar result was also achieved when the model was trained to identify Aedes Aegpyti mosquitoes amongst
other mosquitoe genus belonging to the Culex and Anopheles(the datasets and models were not downloaded into the repository)

The disparity in results likely stemmed from the nature of the datasets, where the different wound types had very similiar characteristics and there is also a lack of variety of images that the model was exposed to. On the other hand, the 
dataset containing happy and sad human faces had a huge pool of images with large variety that allowed the model to better predict the image type.
