# Breast Cancer Classification With Logistic Regression From Scratch

Classifying breast tumors as malignant or benign with logistic regression written in Python and NumPy, trained with gradient descent.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/memuna-javedd/breast-cancer-logistic-regression-from-scratch/blob/main/Breast_Cancer_Detection.ipynb)

## What it does
- Cleans the data: drops the `id` column and an empty trailing column, and encodes the diagnosis as 1 (malignant) and 0 (benign)
- Plots every feature against the diagnosis to explore the data
- Standardizes all 30 features (z-score)
- Implements the sigmoid function, the log loss cost function, the partial derivatives and gradient descent by hand, with no scikit-learn
- Notebook: `Breast_Cancer_Detection.ipynb`

## Dataset
`data.csv` (included in this repo): 569 samples with 30 numeric features computed from tumor images (radius, texture, perimeter, area, smoothness and more, each as mean, standard error and worst value), plus a malignant/benign diagnosis. This is the Breast Cancer Wisconsin (Diagnostic) dataset.

## Training setup
- Learning rate: `0.1`
- Iterations: `1000`
- Weights and bias start at `0`

## Run it
Open the notebook in Google Colab using the badge above. To run it locally, install `numpy`, `pandas` and `matplotlib`, then open it with Jupyter.

## Status
Baseline model. Training works and learns weights for all 30 features. I am still adding the evaluation step and improving it as I learn.

## Planned improvements
- Evaluate the trained model: accuracy, precision, recall and a confusion matrix
- Add a train/test split so the results are measured on unseen data
- Track the cost during training and plot the cost curve
- Replace the loops with vectorized NumPy
- Compare against scikit-learn's LogisticRegression
