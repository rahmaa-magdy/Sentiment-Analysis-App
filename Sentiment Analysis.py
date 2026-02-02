
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

## Preprocess
# Read data set
df = pd.read_csv('test.csv', delimiter=',' , encoding='utf-8')
#rename columns
df.columns = ["review", "label"]
# Separate features (review) and target variable (label)
X = df["review"]
y = df["label"]

## train
# Convert text data to numerical format
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
# Splitting
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state= 1 , stratify=y)
# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# test
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

## evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Function to handle button click event
def analyze_sentiment():
    user_input = input_text.get("1.0", tk.END).strip()
    prediction = predict_sentiment(user_input)
    result_label.config(text=f"Sentiment: {prediction}")
    accuracy_label.config(text=f"Model Accuracy: {accuracy:.2%}")
# ================= GUI ================= #

def analyze_sentiment():
    user_input = input_text.get("1.0", tk.END).strip()
    if not user_input:
        result_label.config(text="Please enter some text", fg="orange")
        return

    prediction = predict_sentiment(user_input)
    result_label.config(
        text=f"Sentiment: {prediction}",
        fg="green" if prediction == "Positive" else "red"
    )
    accuracy_label.config(text=f"Model Accuracy: {accuracy:.2%}")

# main window
window = tk.Tk()
window.title("Sentiment Analysis App")
window.geometry("700x500")
window.configure(bg="#f4f6f8")

# style
style = ttk.Style()
style.theme_use("clam")
style.configure(
    "TButton",
    font=("Segoe UI", 12, "bold"),
    padding=10,
    background="#4CAF50",
    foreground="white"
)
style.map("TButton",
          background=[("active", "#43a047")])

# Header
header = tk.Label(
    window,
    text="Sentiment Analysis Using Machine Learning",
    font=("Segoe UI", 18, "bold"),
    bg="#f4f6f8",
    fg="#333"
)
header.pack(pady=20)

subtitle = tk.Label(
    window,
    text="Enter a review and predict whether it is Positive or Negative",
    font=("Segoe UI", 11),
    bg="#f4f6f8",
    fg="#666"
)
subtitle.pack(pady=5)

# text input
input_text = scrolledtext.ScrolledText(
    window,
    wrap=tk.WORD,
    width=70,
    height=8,
    font=("Segoe UI", 11)
)
input_text.pack(pady=20)

# analyze button
analyze_button = ttk.Button(
    window,
    text="Analyze Sentiment",
    command=analyze_sentiment
)
analyze_button.pack(pady=10)

# result
result_label = tk.Label(
    window,
    text="Sentiment: ---",
    font=("Segoe UI", 16, "bold"),
    bg="#f4f6f8"
)
result_label.pack(pady=10)

accuracy_label = tk.Label(
    window,
    text=f"Model Accuracy: {accuracy:.2%}",
    font=("Segoe UI", 12),
    bg="#f4f6f8",
    fg="#555"
)
accuracy_label.pack(pady=5)

# footer
footer = tk.Label(
    window,
    text="Naive Bayes | NLP | Scikit-learn | Tkinter",
    font=("Segoe UI", 10),
    bg="#f4f6f8",
    fg="#888"
)
footer.pack(side="bottom", pady=15)

window.mainloop()
