# SMS Spam Classification

# Overview:
This project aims to build a machine learning model to classify SMS messages as spam or not spam. The project involves several stages, including data cleaning, Exploratory Data Analysis (EDA), feature extraction using TF-IDF, model training, and evaluation. A Streamlit app is provided to interact with the model in a user-friendly way.

# Project Structure
SMS Spam Classification.ipynb: A Jupyter Notebook that walks through the entire workflow, from data cleaning to model evaluation.
app.py: The Streamlit app file that allows users to input SMS messages and receive predictions (spam or not spam).
vectorizer.pkl: The saved TF-IDF vectorizer used to convert text data into numerical features.
model.pkl: The trained Naive Bayes model used for classifying SMS messages.

# Running the Application
Streamlit App:
To run the Streamlit app, execute the following command in your terminal:
Copy code
streamlit run app.py
This will start a local web server and open the app in your default web browser.
In the app, you can input an SMS message, and the model will predict whether it is spam or not.
Training the Model:

The entire process of training the model is available in the SMS Spam Classification.ipynb notebook. Open it using Jupyter Notebook or Jupyter Lab and run the cells to see the process.

#Workflow
1. Data Cleaning
The dataset is first cleaned to remove any inconsistencies or irrelevant information. This includes:
1) Removing duplicates
2) Handling missing values
3) Normalizing text (e.g., converting to lowercase, removing punctuation)
2. Exploratory Data Analysis (EDA)
EDA is performed to understand the underlying patterns and characteristics of the data. This includes:

Analyzing the distribution of spam vs. non-spam messages
Word frequency analysis to identify common words in spam and non-spam messages
Visualizing the length of messages and other relevant features
3. Visualization
Data visualizations are created to provide insights into the dataset, such as:

Bar charts showing the frequency of spam and non-spam messages
Word clouds to visualize the most common words in spam and non-spam categories
Histograms depicting the distribution of message lengths
4. Feature Extraction
Text features are extracted using the TF-IDF (Term Frequency-Inverse Document Frequency) method. This step converts textual data into numerical features that can be used by the machine learning model.

5. Model Training
The Naive Bayes algorithm is used to train the model on the processed data. The model is evaluated based on various metrics such as accuracy, precision, and recall.

6. Model Evaluation
The performance of the trained model is evaluated using a test dataset. Metrics like accuracy, precision, recall, and F1-score are calculated to assess the model's effectiveness.

7. Saving the Model
The trained model and vectorizer are saved as .pkl files for future use.

# Results
The project demonstrates strong performance in classifying SMS messages, with high accuracy and precision. Detailed performance metrics and visualizations are available in the notebook.
