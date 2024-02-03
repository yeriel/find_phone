# Find Phone

This project involves the implementation of a prototype visual object detection system for a specific client. The goal is to locate a fallen mobile phone on the ground using images captured by a single RGB camera.

## Getting Started

These instructions will provide you with a working copy of the project on your local machine for development and testing purposes. Refer to the [journal]() section for notes on how the project is implemented.

### Prerequisites

To run the project, you need the following libraries as prerequisites.

* Python
* Pytorch
* Jupyterlab

### Installation

Follow this guide to have a functional environment for the project.

Open a terminal

```
$ git clone https://github.com/yeriel/find_phone.git
$ cd find_phone
$ python3 -m venv env
```
Install the dependencies using pip.

```
$ pip install -r requirement.txt
```
Currently, you have your environment ready for development or testing in the project.

## Dataset

The dataset to train the model was provided by Inria and is proprietary. To obtain access, please contact them directly.

#### Structure

the dataset must be composed of a bunch of images and a labels.txt file in the following format img_path x (coordinate of the phone) y (coordinate of the phone) 

Here​ ​is​ ​an​ ​example​ ​of​ ​the​ ​first​ ​3​ ​lines​ 
​from​ ​labels.txt: 

``` 
51.jpg​​ 0.2388​​ 0.6012
95.jpg ​​0.2551​ ​0.3129
84.jpg ​​0.7122 ​​0.7117 
```

## Train model

To train the model run the following command in the terminal 

```
$ p​ython​ t​rain_phone_finder.py​ ~​/path_dataset
```
at the end of the training the weights of the model are saved in the weights folder and in the graphics folder are the loss curves that the model had during the training.

## Inference with the model

To perform an inference of an image with the model you must execute the following script 

```
$ p​ython ​f​ind_phone.py​ ~​/test_images/51.jpg
```
the output of the script corresponds to the (x, y) coordinates of the phone location

```
# output
0.2551​ 0​.3129 
```

## Structure Project  

```
└── 📁find_phone_data       # Dataset
|    └── 📁data
|        └── 10.jpg
|        └── labels.txt
|
└── 📁plots                 # Plots metrics
|    └── Loss_plot.png
|    └── MAE_plot.png
|    └── MSE_plot.png
|
└── 📁utils                 # Tools 
|    └── 📁models
|    |    └── model.py
|    |    └── modules.py
|    └── tools.py
|    └── dataset.py
|    └── trainer.py
|
└── 📁weights               # Weights of model 
|    └── .gitkeep
|    └── best.pth
|
└── search_LR.ipynb         # Search hyperparameters
└── train_phone_finder.py   # Train model
└── find_phone.py           # Inference model
└── requirement.txt         # Dependencies
└── journal.md              # Notes
└── README.md               
```