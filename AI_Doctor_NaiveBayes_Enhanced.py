
# 🩺 AI Doctor - Disease Predictor using Naive Bayes
# -------------------------------------------------
# This project demonstrates a simple AI model that predicts diseases from symptoms using the Naive Bayes theorem.
# It uses a CSV file of symptom-disease pairs to train the model, then predicts based on user input.
# GUI is built using tkinter and the ML model is created using scikit-learn.

import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Initialize model variables
model = None
vectorizer = None
trained = False

# Function to train the model using the selected CSV
def train_model():
    global model, vectorizer, trained
    file_path = filedialog.askopenfilename(
        title="Select Training CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        return
    try:
        # Read the CSV with symptoms and disease
        data = pd.read_csv(file_path)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['symptoms'])  # Convert symptom text to numeric features
        y = data['disease']                             # Target label: disease
        model = MultinomialNB()
        model.fit(X, y)                                 # Train the model
        trained = True
        messagebox.showinfo("Model Trained ✅", "The AI Doctor has been trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {str(e)}")

# Function to predict disease from typed symptoms
def predict_disease():
    global model, vectorizer, trained
    if not trained:
        messagebox.showwarning("Model Not Trained", "⚠️ Please train the model first using a CSV file.")
        return

    symptoms_input = entry.get().strip().lower()
    if not symptoms_input:
        messagebox.showwarning("No Input", "Please enter symptoms before predicting.")
        return

    input_vector = vectorizer.transform([symptoms_input])
    prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    confidence = round(max(probabilities) * 100, 2)

    # Determine color and emoji based on confidence
    if confidence >= 80:
        color = "green"
        emoji = "🩺"
    elif confidence >= 50:
        color = "orange"
        emoji = "🤔"
    else:
        color = "red"
        emoji = "😐"

    # Display prediction result
    result_label.config(
        text=f"{emoji} Prediction: {prediction}\nConfidence: {confidence}%",
        fg=color
    )

# Function to reset everything
def reset_all():
    global model, vectorizer, trained
    trained = False
    model = None
    vectorizer = None
    entry.delete(0, tk.END)
    result_label.config(
        text="🔄 Model and input cleared. Start again from Step 1.",
        fg="black"
    )

# GUI Setup
root = tk.Tk()
root.title("🧠 AI Doctor - Disease Predictor using Naive Bayes")
root.geometry("600x500")
root.configure(bg="#f0f8ff")  # Light blue background

# Header
header = tk.Label(root, text="👨‍⚕️ Welcome to AI Doctor", font=("Arial", 20, "bold"), bg="#f0f8ff", fg="#2c3e50")
header.pack(pady=10)

instructions = tk.Label(
    root,
    text="Step 1: Train the AI using a symptoms CSV file\nStep 2: Enter symptoms to predict disease",
    font=("Arial", 11),
    bg="#f0f8ff",
    fg="#34495e"
)
instructions.pack()

# Input label
input_label = tk.Label(root, text="Enter Symptoms (e.g., 'fever cough'):", font=("Arial", 12, "bold"), bg="#f0f8ff")
input_label.pack(pady=(20, 5))

# Input box
entry = tk.Entry(root, width=60, font=("Arial", 12))
entry.pack(pady=5)

# Result label
result_label = tk.Label(
    root,
    text="🩻 I will show predictions here after training...",
    font=("Arial", 12),
    wraplength=550,
    bg="#f0f8ff"
)
result_label.pack(pady=20)

# Buttons
train_btn = tk.Button(root, text="📂 Train with CSV", command=train_model, bg="#27ae60", fg="white", font=("Arial", 11))
train_btn.pack(pady=5)

predict_btn = tk.Button(root, text="🔍 Predict Disease", command=predict_disease, bg="#2980b9", fg="white", font=("Arial", 11))
predict_btn.pack(pady=5)

reset_btn = tk.Button(root, text="🔁 Reset All", command=reset_all, bg="#c0392b", fg="white", font=("Arial", 11))
reset_btn.pack(pady=5)

# Disclaimer
disclaimer = tk.Label(
    root,
    text="⚠️ This tool is for educational purposes only. It does not provide real medical advice. /nAlways consult a qualified doctor for actual diagnosis and treatment.",
    font=("Arial", 9, "italic"),
    fg="red",
    bg="#f0f8ff",
    wraplength=550,
    justify="center"
)
disclaimer.pack(pady=15)

# Run the GUI loop
root.mainloop()
