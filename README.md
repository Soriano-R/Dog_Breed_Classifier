# Dog Breed Classifier

## Project Overview
This project implements a convolutional neural network (CNN) to classify dog breeds from images. The system can:
- Detect whether an image contains a dog or a human
- If a dog is detected, predict its breed (from 133 possible breeds)
- If a human is detected, suggest the most resembling dog breed

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

### Dataset
This project uses two datasets:

- Dog Dataset: 8,351 dog images of 133 breeds
- Human Dataset: 13,233 human face images

### Project Structure
```
Dog-Breed-Classifier/
│
├── data/
│   ├── dogImages/            # Dog images dataset (train/valid/test)
│   └── lfw/                  # Human images dataset
│
├── saved_models/            # Trained model weights
│   ├── weights.best.VGG16.keras
│   ├── weights.best.Resnet50.keras
│   └── weights.best.from_scratch.keras
│
├── bottleneck_features/     # Pre-computed features from transfer learning
│
├── haarcascades/            # Face detection models
│   └── haarcascade_frontalface_alt.xml
│
├── dog_app.ipynb           # Main Jupyter notebook with code
├── dog_app.html            # HTML export of the notebook
├── extract_bottleneck_features.py  # Helper functions
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Implementation Details
### Model Architecture
The project implements three different CNN approaches:

1. CNN from Scratch: Custom CNN architecture with convolutional and pooling layers
2. VGG16 Transfer Learning: Using pre-trained VGG16 with custom classification layers
3. ResNet50 Transfer Learning: Using pre-trained ResNet50 with custom classification layers

### Key Features
- Face detection using OpenCV's Haar cascades
- Dog detection using pre-trained ResNet50
- Transfer learning from popular CNN architectures
- Data augmentation techniques
- Model checkpointing to save best weights

### Results
- CNN from Scratch: ~10% test accuracy
- VGG16 Transfer Learning: ~72.49% test accuracy
- ResNet50 Transfer Learning: ~80+% test accuracy

The transfer learning approaches significantly outperform the CNN built from scratch, demonstrating the effectiveness of leveraging pre-trained models for specific classification tasks.

Dependencies
- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- OpenCV
- Matplotlib
- PIL/Pillow
- tqdm

### Acknowledgments
- Udacity for the project architecture and datasets
- The creators of VGG16 and ResNet50 architectures

### License
This project is licensed under the MIT License - see the LICENSE file for details.
