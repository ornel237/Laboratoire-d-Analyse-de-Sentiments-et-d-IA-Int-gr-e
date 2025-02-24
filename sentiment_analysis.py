import os
import pandas as pd
import numpy as np
import re
import shap
import pyprind
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from codecarbon import EmissionsTracker

# TASK 1: Data Preparation
basepath = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
data = []
for s in ('test', 'train'):
    for label in ('pos', 'neg'):
        path = os.path.join(basepath, s, label)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
                data.append([txt, labels[label]])
                pbar.update()
df = pd.DataFrame(data, columns=['review', 'sentiment'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

# TASK 2: Text Preprocessing
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    text = text + " " + " ".join(emoticons).replace('-', '')
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)
df['cleaned_review'] = df['review'].apply(preprocess_text)

# TASK 3: Feature Extraction
tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['cleaned_review'])
y = df['sentiment'].values

# TASK 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
pipeline = LogisticRegression(max_iter=10000, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# TASK 5: Model Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# TASK 6: Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 200, 300]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)

# TASK 7: Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(pipeline, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 15), cv=10, n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
plt.plot(train_sizes, test_mean, 'o-', color='green', linestyle='--', label='Validation Accuracy')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# TASK 8: Model Explainability with SHAP
explainer = shap.Explainer(pipeline, X_train)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], feature_names=tfidf.get_feature_names_out())
index = 10
shap.force_plot(explainer.expected_value, shap_values[index].values, feature_names=tfidf.get_feature_names_out(), matplotlib=True)
print("\n🔹 Explication détaillée de la critique sélectionnée 🔹")
print(f"Texte original: {df['review'].iloc[index]}")
sorted_shap_values = sorted(zip(shap_values[index].values, tfidf.get_feature_names_out()), reverse=True, key=lambda x: abs(x[0]))
for i, (value, word) in enumerate(sorted_shap_values[:10]):
    influence = "↗️ Augmente la probabilité positive" if value > 0 else "↘️ Augmente la probabilité négative"
    print(f"{word}: {value:.4f} {influence}")
