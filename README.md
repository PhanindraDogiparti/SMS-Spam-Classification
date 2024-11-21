# SMS Spam Classification

A machine learning project to classify SMS messages as **spam** or **not spam**, with an interactive Streamlit app for real-time predictions. The project covers data cleaning, Exploratory Data Analysis (EDA), feature extraction using TF-IDF, model training, and evaluation.

---

## 📂 Project Structure

- **`SMS Spam Classification.ipynb`**: Jupyter Notebook containing the end-to-end workflow, from data preprocessing to model evaluation.
- **`app.py`**: Streamlit app file for user interaction. Allows you to input an SMS message and get predictions in real time.
- **`vectorizer.pkl`**: Saved TF-IDF vectorizer for converting text into numerical features.
- **`model.pkl`**: Trained Naive Bayes model used for classifying SMS messages.

---

## 🚀 Running the Application

### Streamlit App
Run the Streamlit app using the following command:

```bash
streamlit run app.py


## Here is the updated README.md content with the additional section included in Markdown format:

markdown
Copy code
# SMS Spam Classification

A machine learning project to classify SMS messages as **spam** or **not spam**, with an interactive Streamlit app for real-time predictions. The project covers data cleaning, Exploratory Data Analysis (EDA), feature extraction using TF-IDF, model training, and evaluation.

---

## 📂 Project Structure

- **`SMS Spam Classification.ipynb`**: Jupyter Notebook containing the end-to-end workflow, from data preprocessing to model evaluation.
- **`app.py`**: Streamlit app file for user interaction. Allows you to input an SMS message and get predictions in real time.
- **`vectorizer.pkl`**: Saved TF-IDF vectorizer for converting text into numerical features.
- **`model.pkl`**: Trained Naive Bayes model used for classifying SMS messages.

---

## 🚀 Running the Application

### Streamlit App
Run the Streamlit app using the following command:

```bash
streamlit run app.py
This will launch a local web server and open the app in your default browser.
In the app, input an SMS message, and the model will predict whether it is spam or not.
🛠 Training the Model
The model training process is outlined in the SMS Spam Classification.ipynb notebook. To retrain the model:

Open the notebook in Jupyter Notebook or Jupyter Lab.
Run the cells to follow the process step-by-step.
📊 Workflow Overview
1. Data Cleaning
The dataset is cleaned to prepare it for analysis and modeling:

Removing duplicates
Handling missing values
Normalizing text (e.g., converting to lowercase, removing punctuation)
2. Exploratory Data Analysis (EDA)
EDA provides insights into the dataset:

Distribution of spam vs. non-spam messages
Word frequency analysis
Visualizations (e.g., bar charts, word clouds, histograms)
3. Visualization
Insights are visualized using:

Bar charts: Frequency of spam vs. non-spam messages
Word clouds: Common words in spam and non-spam categories
Histograms: Message length distribution
4. Feature Extraction
Text data is converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) method.

5. Model Training
A Naive Bayes algorithm is trained on the processed data. Evaluation metrics include:

Accuracy
Precision
Recall
F1-score
6. Model Evaluation
The trained model is tested on unseen data to assess its effectiveness. Metrics like accuracy and precision are calculated.

7. Saving the Model
Both the trained model and the TF-IDF vectorizer are saved as .pkl files for future use.

✅ Results
The model achieves high accuracy and performs well in classifying SMS messages. Detailed performance metrics and visualizations are available in the notebook.
