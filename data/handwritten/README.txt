README file for the IHR-NomDB database
Date: November 2020

The handwritten dataset for ChuNom language
There are three folders including 'pages', 'patches' and 'patches_preprocessed'

The Patches folder contains the image and its corresponding annotation for single vertical line text recognition (and translation). The train.json and val.json contains the list of annotation files should be used for training and validating, respectively.
The patches_preprocessed folder is similar with the data of Patches folder, however we have applied the preprocessing steps described on the paper IHR-NomDB: The Old Degraded Vietnamese Handwritten Script Archive Database
The Pages folder contains the original images and annotations, along with its bounding boxes for full page image handwriting recognition, vertical text line localization, start of column dectection, etc.