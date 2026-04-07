import requests
import json

def test_demand(promo, seasonal):
    url = "http://127.0.0.1:8000/predict/demand"
    payload = {
        "data": {
            "price": 30,
            "section": 0, # Women
            "Promotion": promo,
            "Seasonal": seasonal,
            "Product Position_End-cap": 1
        }
    }
    response = requests.post(url, json=payload)
    return response.json()

print("--- TESTING DEMAND MODEL (ISOLATED) ---")
base = test_demand(0, 0)
print(f"Base Case (No Promo): {base['prediction']:.2f} units/week")

boost = test_demand(1, 1)
print(f"Boosted Case (Promo+Seasonal): {boost['prediction']:.2f} units/week")

impact = boost['prediction'] / base['prediction']
print(f"Aggressive Response Factor: {impact:.2f}x")
