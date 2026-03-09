# Titanic Survival Prediction

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## Overview
This project uses a Voting Classifier consisting of a Random Forest Classifier and a XgBoost Classifier to predict the survival of individual present in the Titanic ship. A user can also predict if he/she could have survived the Titanic by providing info in the UI and the model will automatically predict.

![Titanic Model GUI](./Titanic%20Model%20GUI.png)

## Goal
Understand the factors such as sex and age that contributed to the survival of individuals in the Titanic.

## Project Structure
```
Titanic_Model/
├──data/                   # Contains the dataset
│    └── dataset.csv       # Titanic dataset used in training
├──models/                 # Contains the saved pipeline
│    └──model_pipeline.pkl # Pipeline for predicting titanic survival
├──src
│    ├──app.py             # Streamlit UI for titanic survival    
│    └──trainer.py         # Used in training the model on the dataset
└── README.md              # This file
```

## Dataset
Dataset is obtained from Kaggle. It contains 891 rows and 12 columns. The dataset is then split into training and testing sets during training. 

## Prerequisites
Before cloning or using the code ensure you have:

1. Python 3.10 or higher
2. pip (Python Package Installer)
3. The following Python packages (installed via `requirements.txt`):
   - `numpy`, `pandas`, `streamlit`, `scikit-learn`, `xgboost-cpu`, `scipy`

# Installation

## Clone this repo
```bash
git clone <your-repository-url>
cd Titanic_Model
```
## Install dependencies
```bash
pip install -r requirements.txt
```

# How to use :

## Usage :
```bash
streamlit run src/app.py
```
##### NB : Follow the Link provided if it does not open 

## Model Training
1. The dataset is split into training and testing set in a ratio of 4:1
2. The model is trained using a Voting Classifier consisting of a Random Forest Classifier and a XgBoost Classifier
3. The model is then saved to a file named model_pipeline.pkl
4. The model is then used to predict the survival of individuals in the Titanic

### Train the model using :
```bash
python src/trainer.py
```

## Results
1. Accuracy : 0.86
2. Precision : 0.77
3. Mean Squared Error : 0.12

## Contributing
All contributions are welcome. Please open issues or submit pull requests.

## License
MIT License
