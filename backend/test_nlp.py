import requests
import json

def test_nlp(text):
    url = "http://127.0.0.1:8000/predict/nlp"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    return response.json()

print("--- TESTING NLP MODEL (ISOLATED) ---")
pos = test_nlp("This dress is amazing! The quality is great.")
print(f"Positive Test: {pos['sentiment']} (Score: {pos['sentiment_score']})")

neg = test_nlp("Horrible quality. The fabric is cheap and the size is wrong.")
print(f"Negative Test: {neg['sentiment']} (Score: {neg['sentiment_score']})")
