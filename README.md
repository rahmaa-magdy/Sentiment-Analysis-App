# Sentiment-Analysis-App
A machine learning–based desktop application that predicts the sentiment of text reviews (Positive / Negative) using Natural Language Processing techniques and a Multinomial Naive Bayes classifier.

---

Project Overview
This project aims to:
- Classify text reviews into positive or negative sentiments
- Apply NLP techniques to transform raw text into numerical features
- Train and evaluate a machine learning classification model
- Provide an interactive desktop GUI for real-time sentiment prediction

---

Technologies Used
- Python
- Pandas
- Scikit-learn
- Tkinter
- NLP (Bag-of-Words)

---

Dataset
- File: `test.csv`
- Columns:
  - `review` → text review
  - `label` → sentiment label (1 = Positive, 0 = Negative)

---

Data Preprocessing
- Renamed dataset columns for clarity
- Separated features and labels
- Converted text data into numerical format using CountVectorizer
- Removed English stop words to improve model performance

---

Model Training
- Algorithm: Multinomial Naive Bayes
- Feature Extraction: Bag-of-Words
- Train-Test Split: 80% training, 20% testing (stratified)
- Random State set for reproducibility

---

Model Evaluation
- Metric used: Accuracy
- The trained model evaluates sentiment predictions on unseen test data
- Accuracy is displayed dynamically in the GUI

---

Graphical User Interface (GUI)
- Built using Tkinter
- Allows users to:
  - Enter custom text reviews
  - Predict sentiment instantly
  - View model accuracy
- Color-coded prediction results (Green = Positive, Red = Negative)

