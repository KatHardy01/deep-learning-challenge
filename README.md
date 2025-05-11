# Deep Learning Challenge

## Introduction 
Alphabet Soup, a nonprofit foundation, aims to identify applicants for funding who are most likely to succeed in their ventures. Using historical data of over 34,000 organizations, this project builds a binary classification model to predict whether an applicant will be successful if funded. The goal is to assist Alphabet Soup in making data-driven funding decisions.

This project uses a deep learning neural network to analyze the dataset and attempts to optimize the model to achieve an accuracy of 75% or higher.

## Challenge Instructions: 
**Step 1: Import Dependencies and Load Dataset**
* Include all necessary libraries (e.g., `pandas`, `tensorflow`, `sklearn`) and load the dataset into a DataFrame.

**Step 2: Inspect and Preprocess the Data**
* Drop Non-Beneficial Columns: Remove columns like `EIN`, `NAME`, and others that do not contribute to the prediction.
* Analyze Unique Values in Categorical Columns: Identify columns with many unique values to determine if rare categories need to be grouped.
* Combine Rare Categories: Group rare categories into an "Other" category for columns like `APPLICATION_TYPE` and `CLASSIFICATION`.
* Encode Categorical Variables: Use one-hot encoding (`pd.get_dummies`) to convert categorical variables into numerical format.
* Split Data into Features and Target: Separate the target variable (`IS_SUCCESSFUL`) from the features.
* Scale Features: Use `StandardScaler` to normalize numerical features for better model performance.

**Step 3: Build and Train the Neural Network**
* Define the Model Architecture: Create the initial neural network with input, hidden, and output layers.
* Compile and Train the Model: Compile the model with an optimizer (e.g., Adam), loss function (e.g., binary crossentropy), and evaluation metric (e.g., accuracy). Train the model on the training dataset.
* Save Model Weights with Callbacks: Use callbacks like `ModelCheckpoint` to save the model weights during training.
* Evaluate Model Performance: Evaluate the model on the test dataset to calculate accuracy and loss.

**Step 4: Optimize the Model**
* Create a new notebook to train and experiment models to optimize 
* Adjust Input Data: Experiment with dropping additional columns, combining categories, or applying transformations.
* Modify Neural Network Architecture: Add more layers, neurons, Batch Normalization, or Dropout to improve performance.
* Retrain and Evaluate the Optimized Model

**Step 5: Save Final Model**
* Save the final trained model to an HDF5 file

**Step 6: Write A Report**
* Summarize the results, challenges, and recommendations for future improvements.


## Report on the Neural Network Model and Results

### **Overview of the Analysis**
The purpose of this analysis is to create a binary classification model using a deep learning neural network to predict whether applicants for funding from Alphabet Soup will be successful in their ventures. By analyzing historical data, the model aims to assist Alphabet Soup in making data-driven decisions to allocate funding to applicants with the highest likelihood of success.

### **Results**

Model 1: used omptimizer Adamax and activation LeakyReLu. Accuracy: 73.6% and loss .556.
Model 2: used actication function Tanh. Accuracy: 73.6% and loss .548.
Model 3: Random forest classifier. Accuracy: 72.1%

### Summary
Overall Results: The deep learning model achieved similar accuracy across all three models. The highest accuracy was 73.6% which is below the target of 75%. Despite extensive optimizations, the model struggled to extract enough meaningful patterns from the data to achieve the desired performance.
