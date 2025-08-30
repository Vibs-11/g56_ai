# ==================== LESSON MAPPING & ADDITIONAL ACTIVITIES ====================
# This file is the SINGLE CAPSTONE (Lessons 13–23) for the AI Doctor project.
# Per requirement: We DO NOT modify existing functionality; we only add
# - lesson-wise comments (including '### LESSON XX COMPLETE' markers), and
# - additional activity code as separate helper demos (CLI), kept non-invasive.
#
# HOW TO LOCATE LESSONS (markers appear near the relevant code):
#   • LESSON 13  → Intro to VS Code & Supervised Learning        (see CLI demos)
#   • LESSON 14  → Create symptoms.csv dataset                   (see CLI demos)
#   • LESSON 15  → Train model with Naive Bayes (train_model)    (marker in fn)
#   • LESSON 16  → Predict disease from text (predict_disease)   (marker in fn)
#   • LESSON 17  → Confidence % + color tags                     (marker in fn)
#   • LESSON 18  → Explain prediction with matched words         (see CLI demos)
#   • LESSON 19  → Improve with new data & compare accuracy      (see CLI demos)
#   • LESSON 20  → Text-to-Speech / simulated voice              (see CLI demos)
#   • LESSON 21  → Tkinter GUI (main UI code below)              (marker below)
#   • LESSON 22  → Reset button + disclaimer                     (marker below)
#   • LESSON 23  → GitHub README + troubleshooting builder       (see CLI demos)
#
# The function print_marker_index() at the bottom can print the ACTUAL line
# numbers of each '### LESSON XX COMPLETE' marker in this file.
# ===============================================================================


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
        # ### LESSON 15 COMPLETE — Model training finished (Naive Bayes fit)
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
    # ### LESSON 16 COMPLETE — Printed predicted disease
    # ### LESSON 17 COMPLETE — Displayed confidence %, bar/color via tags

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
    # ### LESSON 22 COMPLETE — Reset action and safety formatting present

# GUI Setup
root = tk.Tk()
root.title("🧠 AI Doctor - Disease Predictor using Naive Bayes")  # ### LESSON 21 COMPLETE — Basic Tkinter GUI ready
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
    text="⚠️ This tool is for educational purposes only. It does not provide real medical advice. \nAlways consult a qualified doctor for actual diagnosis and treatment.",
    font=("Arial", 9, "italic"),
    fg="red",
    bg="#f0f8ff",
    wraplength=550,
    justify="center"
)
disclaimer.pack(pady=15)

# Run the GUI loop
root.mainloop()

# ==================== ADDITIONAL ACTIVITIES (CLI DEMOS) ====================
# These functions implement the extra activities for Lessons 13, 14, 15, 16, 17, 18,
# 19, 20, 21(text fallback), 22(text menu variant), and 23 — without changing the GUI.

def aa_l13_examples_list():
    """
    Lesson 13 Additional Activity (CLI):
      - Print title
      - Collect 5 pairs (input → label) using input()
      - Store in two lists (inputs, labels)
      - Print a neat table, then ask for 1 new input and exact-match predict
      - Print 3 lines about how examples teach a computer
    """
    print("=== Examples List (L13) ===")
    inputs, labels = [], []
    for i in range(1, 6):
        pair = input(f"Pair {i} (e.g., dog, woof): ").strip()
        if "," in pair:
            a, b = pair.split(",", 1)
            inputs.append(a.strip())
            labels.append(b.strip())
    print("\nNo. | Input        | Label")
    print("-" * 26)
    for i, (a, b) in enumerate(zip(inputs, labels), 1):
        print(f"{i:>2}  | {a:<12} | {b}")
    q = input("\nTry a new input to predict: ").strip()
    if q in inputs:
        idx = inputs.index(q)
        print("Prediction:", labels[idx])
    else:
        print("Prediction: Not sure yet")
    print("-" * 40)
    print("Learning note 1: We gave examples as input → correct output.")
    print("Learning note 2: The computer can use examples to guess for new inputs.")
    print("Learning note 3: More examples usually improve guesses.")
    # ### LESSON 13 COMPLETE — CLI activity finished

def aa_l14_collect_in_memory():
    """
    Lesson 14 Additional Activity (CLI): collect 10 'symptoms, disease' rows
    into two lists and preview unique diseases.
    """
    print("=== Build In-Memory Dataset (L14) ===")
    symptoms_list, disease_list = [], []
    for i in range(1, 11):
        row = input(f"Row {i} (symptoms, disease): ")
        if "," not in row: 
            print("Please include a comma; try again."); continue
        s, d = row.split(",", 1)
        symptoms_list.append(s.strip()); disease_list.append(d.strip())
    print("\nNo. | Symptoms                 | Disease")
    print("-" * 44)
    for i, (s, d) in enumerate(zip(symptoms_list, disease_list), 1):
        print(f"{i:>2}  | {s[:22]:<22} | {d}")
    uniques = []
    for d in disease_list:
        if d not in uniques: uniques.append(d)
    print(f"\nTotal rows: {len(symptoms_list)} | Unique diseases: {len(uniques)}")
    print("Summary: Collected dataset in memory (lists only).")
    # ### LESSON 14 COMPLETE — CLI activity finished

def aa_l15_pretend_training_counts():
    """
    Lesson 15 Additional Activity (CLI): pretend training by word counts.
    """
    print("=== Pretend Training by Word Counts (L15) ===")
    sample_sym = []
    for i in range(6):
        row = input(f"Sample {i+1} (symptoms, disease): ")
        if "," in row:
            s, d = row.split(",", 1)
            sample_sym.append(s.strip().lower())
    words = []
    for s in sample_sym: words += s.split()
    interesting = ["fever", "cough", "headache", "rash"]
    for w in interesting:
        print(f"{w}: {sum(1 for x in words if x == w)}")
    print("Training means learning counts/patterns from examples.")
    # ### LESSON 15 COMPLETE — CLI activity finished

def aa_l16_predict_match_score():
    """
    Lesson 16 Additional Activity (CLI): simple word-match score predictor.
    """
    print("=== Match-Score Predictor (L16) ===")
    pairs = []
    for i in range(6):
        row = input(f"Row {i+1} (symptoms, disease): ")
        if "," in row:
            s, d = row.split(",", 1)
            pairs.append((s.strip().lower(), d.strip()))
    def score(user, row_sym):
        uw = user.split(); rw = row_sym.split()
        return sum(1 for w in uw if w in rw)
    for t in range(2):
        user = input("Type symptoms: ").strip().lower()
        best_dis, best = "Not sure", 0
        for s, d in pairs:
            sc = score(user, s)
            if sc > best: best, best_dis = sc, d
        print("Predicted:", best_dis)
    # ### LESSON 16 COMPLETE — CLI activity finished

def aa_l17_confidence_bar():
    """
    Lesson 17 Additional Activity (CLI): compute best/total confidence and show bar+tag.
    """
    print("=== Confidence Bar (L17) ===")
    pairs = []
    for i in range(6):
        row = input(f"Row {i+1} (symptoms, disease): ")
        if "," in row:
            s, d = row.split(",", 1)
            pairs.append((s.strip().lower(), d.strip()))
    def score(user, row_sym):
        uw = user.split(); rw = row_sym.split()
        return sum(1 for w in uw if w in rw)
    for t in range(2):
        user = input("Type symptoms: ").strip().lower()
        scores = [score(user, s) for s, _ in pairs]
        total = sum(scores); best_score = max(scores) if scores else 0
        best_dis = pairs[scores.index(best_score)][1] if best_score>0 else "Not sure"
        conf = (best_score/total)*100 if total>0 else 0
        bar = "#"*int(round(conf/10)) + "-"*(10-int(round(conf/10)))
        tag = "[HIGH]" if conf>=70 else "[MEDIUM]" if conf>=40 else "[LOW]"
        print(f"Prediction: {best_dis} | Confidence: {conf:.1f}% {bar} {tag}")
    # ### LESSON 17 COMPLETE — CLI activity finished

def aa_l18_explain_prediction():
    """
    Lesson 18 Additional Activity (CLI): show matched words, not matched, emoji legend.
    """
    print("=== Explain Prediction (L18) ===")
    pairs = []
    for i in range(6):
        row = input(f"Row {i+1} (symptoms, disease): ")
        if "," in row:
            s, d = row.split(",", 1)
            pairs.append((s.strip().lower(), d.strip().lower()))
    legend = {"flu":"😷","cold":"🤧","stress":"🤕"}
    def score(user, row_sym):
        uw = user.split(); rw = row_sym.split()
        return sum(1 for w in uw if w in rw)
    for t in range(2):
        user = input("Type symptoms: ").strip().lower()
        best_dis, best_sc = "not sure", 0
        best_row = ""
        for s, d in pairs:
            sc = score(user, s)
            if sc>best_sc: best_sc, best_dis, best_row = sc, d, s
        uw = user.split(); rw = best_row.split()
        found = [w.upper() for w in uw if w in rw]; not_found = [w for w in uw if w not in rw]
        print("User words:", uw)
        print("FOUND:", found)
        print("NOT FOUND:", not_found)
        print("Best disease:", best_dis, legend.get(best_dis, ""))
    print("This is not medical advice.")
    # ### LESSON 18 COMPLETE — CLI activity finished

def aa_l19_before_after_accuracy():
    """
    Lesson 19 Additional Activity (CLI): compare accuracy before/after more data.
    """
    print("=== Before vs After Accuracy (L19) ===")
    base = [("fever cough","flu"),("runny nose","cold"),("headache stress","stress"),
            ("rash itch","allergy"),("stomach pain","food_poisoning"),("high fever rash","dengue")]
    extra = []
    for i in range(6):
        row = input(f"Extra {i+1} (symptoms, disease): ")
        if "," in row:
            s, d = row.split(",", 1)
            extra.append((s.strip().lower(), d.strip().lower()))
    def score(user, row_sym):
        uw = user.split(); rw = row_sym.split()
        return sum(1 for w in uw if w in rw)
    def round_acc(pairs, title):
        print(title); correct = 0
        for t in range(3):
            user = input("Test symptoms: ").strip().lower()
            best_dis, best_sc = "not sure", 0
            for s, d in pairs:
                sc = score(user, s)
                if sc>best_sc: best_sc, best_dis = sc, d
            print("Predicted:", best_dis)
            if input("Was this correct? (y/n): ").strip().lower()=="y": correct+=1
        print("Accuracy:", correct, "/3"); return correct
    a = round_acc(base, "Round A (seed 6)")
    b = round_acc(base+extra, "Round B (seed + added 6)")
    print(f"Conclusion: Round A={a}/3 vs Round B={b}/3. More data often helps.")
    # ### LESSON 19 COMPLETE — CLI activity finished

def aa_l20_tts_or_sim():
    """
    Lesson 20 Additional Activity (CLI): TTS with pyttsx3 or simulate voice.
    """
    try:
        import pyttsx3
    except Exception:
        pyttsx3 = None
    sent = input("Enter a prediction sentence: ")
    rate = input("Choose rate (slow/normal/fast): ").strip().lower()
    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 140 if rate=="slow" else 220 if rate=="fast" else 180)
            print("SPEAK:", sent); engine.say(sent); engine.runAndWait()
        except Exception:
            print("TTS engine not available, using simulation.")
            pyttsx3 = None
    if pyttsx3 is None:
        if rate=="slow":
            print("SPEAK (slow):"); [print(w) for w in sent.split()]
        elif rate=="fast":
            print("SPEAK (fast): >>", sent, ">>", sent)
        else:
            print("SPEAK (normal):", sent)
    print("How TTS works (simple): text → phonetics → audio waveform.")
    # ### LESSON 20 COMPLETE — CLI activity finished

def aa_l23_readme_and_troubleshooting():
    """
    Lesson 23 Additional Activity (CLI): README preview + troubleshooting.
    """
    print("=== README & Troubleshooting Builder (L23) ===")
    title = input("Project Title: ")
    d1 = input("Short description line 1: ")
    d2 = input("Short description line 2: ")
    files = [input("File 1: "), input("File 2: "), input("File 3: ")]
    print("\\n# ", title)
    print("## How to Run\\npython AI_Doctor_NaiveBayes_Enhanced.py")
    print("## What it does\\n-", d1, "\\n-", d2)
    print("## Files:"); [print("-", f) for f in files]
    for _ in range(2):
        print("\\nProblems: 1) File not found  2) Wrong input format  3) Program exited early")
        ch = input("Choose (1/2/3): ").strip()
        tips = {"1":"Check file name & folder path.",
                "2":"Type symptoms like: fever cough",
                "3":"Read the last error message and retry step-by-step."}
        print("Tip:", tips.get(ch,"Read the error text carefully."))
    print("\\nGitHub helps you: share code, track changes, collaborate.")
    # ### LESSON 23 COMPLETE — CLI activity finished

def print_marker_index():
    #\"\"\"Scan this file and print actual line numbers for all LESSON COMPLETE markers.\"\"\"
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print("\\nLesson completion markers (line numbers):")
        for i, text in enumerate(lines, start=1):
            if text.strip().startswith("### LESSON ") and text.strip().endswith("COMPLETE —") or \
               text.strip().startswith("# ### LESSON ") and "COMPLETE" in text:
                print(f"{text.strip()}  -> line {i}")
        print()
    except Exception as e:
        print("Could not read source to list markers:", e)
# ==============================================================================
