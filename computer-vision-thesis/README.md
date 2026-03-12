Synthetic Dataset Generation for Underwater Fish Detection
This MATLAB script was developed as part of a Bachelor's Thesis (TFG) on underwater object detection using deep learning. It generates a fully labeled synthetic training dataset by compositing fish images onto real underwater backgrounds, with automatic bounding box annotation in YOLO format.
Overview
A key challenge in training object detectors for underwater scenes is the scarcity of labeled data. This script addresses that by generating synthetic training samples: a fish (rendered from a clean image with its segmentation mask) is procedurally augmented and blended onto a real underwater background. Because the fish placement is fully controlled, ground-truth bounding boxes can be computed automatically from the mask, eliminating manual annotation entirely.
How It Works
1. Input Data
The script expects the following folder structure under a base path:
path/
├── Background/          # Real underwater background images (.jpg)
├── Generated/
│   └── BlueTang/
│       ├── fish1.jpg    # Fish images (indices 1-10)
│       ├── maskfish1.jpg
│       └── ...
├── referencecolor.jpg   # Reference image for color normalization
└── traindataset/        # Output: final images + YOLO annotations
└── traindatasetvisual/  # Output: images with bounding boxes drawn
Each background .jpg has an associated .mat file containing illumination parameters (params) used for color matching.
2. Image Generation Pipeline (processImage)
For each generated sample, the following augmentation steps are applied to the fish:
StepDetailsMask preprocessingConvert to binary, dilate (r=17px disk) to remove edge artifactsResizeRandom fish size between 20% and 75% of 512pxColor adjustment50% probability: apply illumination-matched imadjust using background paramsRotationRandom angle in [-45, 45] degreesHorizontal flip12.5% probabilityVertical flip50% probabilitySqueeze/stretch25% probability per axis: random scale factor in [0.2, 1.0]NoiseGaussian noise (amplitude ~10-20) + Gaussian blur (sigma=1) + second noise pass (amplitude ~2-5)
The background is also randomized: a crop of 512-1024px is sampled from a random region of the full-resolution (1080x1920) background, then resized to 512x512.
The fish is then placed at a random position within the background. Edge blending is achieved by applying Gaussian blur (sigma=2) to the binary mask before alpha compositing.
3. Automatic Annotation
Bounding boxes are derived directly from the fish mask using regionprops, selecting the largest connected region. The box is expressed in YOLO format:
<class_index> <x_center> <y_center> <width> <height>
All values are normalized to [0, 1] relative to the 512x512 image. Class index is always 0 (single-class detector).
4. Output Files
For each generated image, three files are written:

traindataset/<name>-fish<id>-<timestamp>.jpg — training image
traindataset/<name>-fish<id>-<timestamp>.txt — YOLO annotation
traindatasetvisual/<name>-fish<id>-<timestamp>.jpg — image with bounding box overlay (yellow rectangle, for visual inspection)

Filenames include a UTC millisecond timestamp to avoid collisions.
Usage
matlabgeneratedataset(numimgs)

numimgs: number of synthetic training images to generate

Example:
matlabgeneratedataset(500)
Dependencies

MATLAB Image Processing Toolbox (imread, imresize, imadjust, imgaussfilt, imdilate, imrotate, regionprops, insertShape)

Context
This script was part of the data preparation pipeline for training a YOLOv8 model for underwater object detection. The synthetic generation approach allows rapid dataset scaling without manual labeling, and the domain randomization (lighting, scale, orientation, noise) improves model robustness to real underwater conditions.

Related publication: Generative Model-Driven Underwater Object Detection, University of Ljubljana Machine Vision Laboratory.
