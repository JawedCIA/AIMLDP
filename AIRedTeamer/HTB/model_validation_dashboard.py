# model_validation_dashboard.py

# Install required packages (uncomment and run this line if needed)
# import os
# os.system("pip install fastapi uvicorn requests nest_asyncio")

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dummy test data for validation (replace with actual dataset)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
X_data = [
    "Congratulations! You have won a prize!",
    "Meeting at 10am tomorrow",
    "Free entry in a weekly contest!",
    "Lunch with team today",
    "Win a brand new iPhone now"
]
y_data = [1, 0, 1, 0, 1]  # 1=Spam, 0=Ham

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_data, test_size=0.4, random_state=42)

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_model(request: Request, model: UploadFile = File(...)):
    model_path = f"uploaded_model.joblib"
    with open(model_path, "wb") as f:
        content = await model.read()
        f.write(content)

    clf = load(model_path)

    # Use full dataset instead of tiny test set
    X_test_raw = X_data
    y_test_full = y_data

    try:
        y_pred = clf.predict(X_test_raw)
    except Exception as e:
        return HTMLResponse(f"<h3>Error during prediction: {str(e)}</h3>")

    acc = accuracy_score(y_test_full, y_pred)
    cls_report = classification_report(y_test_full, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test_full, y_pred)

    precision = cls_report.get("weighted avg", {}).get("precision", 0.0)
    recall = cls_report.get("weighted avg", {}).get("recall", 0.0)
    f1_score = cls_report.get("weighted avg", {}).get("f1-score", 0.0)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    image_path = "static/conf_matrix.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(image_path)
    plt.close()

    report_df = pd.DataFrame(cls_report).T.round(4)
    html_table = report_df.to_html(classes="styled-table")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "accuracy": f"{acc:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1_score": f"{f1_score:.2f}",
        "metrics_table": html_table,
        "conf_matrix_img": f"/{image_path}"
    })