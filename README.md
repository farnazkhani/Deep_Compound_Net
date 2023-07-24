# DeepCompoundNet: Compound-Protein Interaction Prediction
## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Project Description

DeepCompoundNet is a deep learning model for predicting compound-protein interactions by utilizing molecular data, compound-compound, protein-protein similarity, and convolutional neural networks. The goal of this project is to computationally predict the existence or absence of interactions between compounds and proteins, providing an efficient and low-cost virtual screening tool for seed molecules.

## Installation

To use DeepCompoundNet, follow these steps:

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd DeepCompoundNet
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To predict compound-protein interactions using DeepCompoundNet, you need to import the code and use it in your Python code. Here's an example of how to use the model:

```python
# Load your compound and protein data
compound_data = load_compound_data('path/to/compound_data.csv')
protein_data = load_protein_data('path/to/protein_data.csv')

# Preprocess the data (if necessary)
preprocessed_data = preprocess_data(compound_data, protein_data)

# Use DeepCompoundNet to predict interactions
predictions = deepcompoundnet.predict(preprocessed_data)

# Display the predictions or save them to a file
save_predictions(predictions, 'path/to/predictions.csv')
```

Make sure to replace `'path/to/compound_data.csv'`, `'path/to/protein_data.csv'`, and `'path/to/predictions.csv'` with the actual file paths.

## Features

DeepCompoundNet offers the following features:

- Integration of molecular structure data and interactome data for improved prediction accuracy.
- Utilization of convolutional neural networks to capture complex patterns in compound-protein interactions.
- Efficient virtual screening of compound-protein interactions not observed in the training samples.

## Technologies Used
Python 
TensorFlow
Pandas 
NumPy
Matplotlib 

Thank you for your interest in DeepCompoundNet! We hope this model will be helpful in your compound-protein interaction prediction tasks. If you have any questions or feedback, please don't hesitate to contact us. Happy predicting!
