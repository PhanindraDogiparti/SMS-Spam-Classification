# SMS Spam Classification

## Overview  
This project builds a machine learning model to classify SMS messages as either spam or not spam. The project pipeline includes data cleaning, Exploratory Data Analysis (EDA), feature extraction with TF-IDF, model training using Naive Bayes, and evaluation. A **Streamlit app** is provided to interact with the model easily.

## Project Structure
- **`SMS Spam Classification.ipynb`**: Jupyter Notebook that walks through the entire workflow, from data cleaning to model evaluation.  
- **`app.py`**: Streamlit app that allows users to input SMS messages and receive predictions (spam or not spam).  
- **`vectorizer.pkl`**: Saved TF-IDF vectorizer for converting text data into numerical features.  
- **`model.pkl`**: Trained Naive Bayes model used to classify SMS messages.





2. Training the Model
The entire process of training the model is available in the SMS Spam Classification.ipynb notebook. Open it in Jupyter Notebook or Jupyter Lab and run the cells to see the process, which includes:

Data cleaning and preprocessing
Feature extraction with TF-IDF
Model training and evaluation
Workflow
1. Data Cleaning
The dataset is cleaned by:

Removing duplicates
Handling missing values
Normalizing text (e.g., converting to lowercase and removing punctuation)
2. Exploratory Data Analysis (EDA)
EDA is conducted to gain insights into the data, including:

Analyzing the distribution of spam vs. non-spam messages
Word frequency analysis for both categories
Visualizing message lengths and other key features
3. Feature Extraction
Text features are extracted using the TF-IDF (Term Frequency-Inverse Document Frequency) method, converting text into numerical features that can be used by machine learning algorithms.

4. Model Training
The Naive Bayes algorithm is used to train the model on the processed data. Evaluation is done based on metrics such as accuracy, precision, recall, and F1-score.

5. Model Evaluation
The trained model is evaluated using a test dataset. Key performance metrics (accuracy, precision, recall, F1-score) are calculated to assess its effectiveness.

6. Saving the Model
The trained model and TF-IDF vectorizer are saved as .pkl files for future use and deployment.

Results
The model demonstrates strong performance in classifying SMS messages with high accuracy and precision. Detailed evaluation metrics and visualizations are included in the Jupyter Notebook.

## Requirements
Python 3.x
Streamlit
scikit-learn
pandas
numpy
matplotlib
wordcloud (for visualization)


## How to Run the Application
### 1. **Streamlit App**  
To run the app, execute the following command in your terminal:  
```bash
streamlit run app.py
This will start a local web server and open the app in your browser. You can then input an SMS message, and the model will predict whether it is spam or not.
