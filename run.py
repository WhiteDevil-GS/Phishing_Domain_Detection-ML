import requests
import Feature_extraction_ff1 as fex  
import numpy as np
import os
import joblib

model_path = "our_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

def preprocess_url(domain):
    if "." not in domain:
        raise ValueError("Invalid URL format")
    if not domain.startswith("http://") and not domain.startswith("https://"):
        domain = "https://" + domain
    if domain.startswith("https://www."):
        domain = "https://" + domain[12:]
    elif domain.startswith("http://www."):
        domain = "http://" + domain[11:]
    
    return domain

def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        raise ConnectionError(f"Error fetching URL: {e}")

def extract_features(domain):
    features = fex.data_set_list_creation(domain)
    if features is None or not isinstance(features, list):
        raise ValueError("Feature extraction failed or returned invalid data")
    return features

def predict_phishing(features):
    if not features:
        raise ValueError("Features are empty or None")
    features = np.array([features])
    prediction = model.predict(features)
    if prediction is None:
        raise ValueError("Prediction failed or returned None")
    
    return prediction[0] == 1

domain_input = "google.com"

try:
    cleaned_domain = preprocess_url(domain_input)
    print(f"Cleaned domain: {cleaned_domain}")
    fetch_url_content(cleaned_domain)
    print(f"URL fetched successfully: {cleaned_domain}")
    features = extract_features(cleaned_domain)
    print(f"Extracted features: {features}")
    is_phishing = predict_phishing(features)
    if is_phishing:
        result = "The URL is predicted as a phishing URL."
    else:
        result = "The URL is predicted as a legitimate URL."
    
    print(result)

except (ConnectionError) as e:
    print(f"Error:")
