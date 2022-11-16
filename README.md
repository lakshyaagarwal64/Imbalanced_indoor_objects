# Imbalanced_indoor_objects
This model concerns multiclass object detection on an imbalanced dataset. The imbalanced dataset concerns indoor objects, and the model is constructed using Resnet-50 as feature extraction layer along with fully connected + Conv layers for bounding box prediction.
The first requirement is to download the data from "https://zenodo.org/record/2654485/files/Indoor%20Object%20Detection%20Dataset.zip?download=1". Then convert the xml files to csv using a convertor.
Following this, split the files into train and test, and make sure all csv files have the same format, namely : file_name, object_1, object_2.. etc. 
Specify the correct file-paths in the dataloader and run the training file to train the model. Heavy data augmentation is used to obtain respectable accuracy on all classes.
