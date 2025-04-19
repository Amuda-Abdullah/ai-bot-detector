import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "Training_Essay_Data.csv"
df = pd.read_csv(file_path)

# Function to clean text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    return text

df['text'] = df['text'].astype(str).apply(clean_text)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['generated'], test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
with open("text_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("model.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Predict on test set
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Load and use the model
def predict_text(text):
    with open("text_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open("model.pkl", "rb") as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    
    text_tfidf = loaded_vectorizer.transform([clean_text(text)])
    prediction = loaded_model.predict(text_tfidf)
    return "Bot-generated" if prediction[0] == 1 else "Human-written"

# Example usage
computer_content = """Hello Nigerians, today is a very good day"""

print(f"Prediction: {predict_text(computer_content)}")
