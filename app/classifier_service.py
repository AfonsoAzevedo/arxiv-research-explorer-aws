"""
Load the trained classifier and predict arXiv research categories.

The model uses TF-IDF and Logistic Regression and was trained
on 40,000 research abstracts using Amazon EC2.
"""

from pathlib import Path

import joblib
import numpy as np


# Locate the production model in the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "classification/models/full-40000-seed42/classifier.joblib"
)

# Map the eight arXiv category codes to readable names.
CATEGORY_NAMES = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision",
    "cs.DB": "Databases",
    "cs.LG": "Machine Learning",
    "cs.NI": "Networking and Internet Architecture",
    "cs.RO": "Robotics",
}


def load_classifier():
    """Load the saved production classifier without retraining it."""

    # Check that the model file exists before loading it.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Production classifier not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def classify_abstract(model, abstract):
    """Predict the research category and return the top three results."""

    # Reject empty inputs before running the model.
    if not isinstance(abstract, str) or not abstract.strip():
        raise ValueError("Please enter a research abstract.")

    # Generate predictions using the saved TF-IDF and Logistic Regression pipeline.
    probabilities = model.predict_proba([abstract])[0]
    classes = model.classes_

    # Find the three categories with the highest model probabilities.
    top_indices = np.argsort(probabilities)[::-1][:3]
    predicted_code = str(classes[top_indices[0]])

    top_categories = []

    for index in top_indices:
        code = str(classes[index])

        top_categories.append({
            "code": code,
            "name": CATEGORY_NAMES[code],
            "probability": float(probabilities[index]),
        })

    # Return the prediction in a format Streamlit can display.
    return {
        "category_code": predicted_code,
        "category_name": CATEGORY_NAMES[predicted_code],
        "top_categories": top_categories,
    }
