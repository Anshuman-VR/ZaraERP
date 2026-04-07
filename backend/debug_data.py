import pandas as pd
import numpy as np
import os

BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP"
DATA_PATH = os.path.join(BASE, "data", "zaraSales.csv")

# Load and Inspect
df = pd.read_csv(DATA_PATH, sep=';')

# Basic Stats
print("--- Sales Volume Stats ---")
print(df["Sales Volume"].describe())
print(f"90th percentile: {df['Sales Volume'].quantile(0.9)}")
print(f"95th percentile: {df['Sales Volume'].quantile(0.95)}")
print(f"Max: {df['Sales Volume'].max()}")

# Synthetic Calculation Test (Recalibrating for 12-week life)
def test_weekly(vol, promo, seasonal):
    base_weeks = 12.0
    if seasonal == 1: base_weeks = 6.0
    if promo == 1: base_weeks *= 0.6
    return vol / base_weeks

df["Weekly_Base"] = df.apply(lambda r: test_weekly(r["Sales Volume"], 0, 0), axis=1)
df["Weekly_Boost"] = df.apply(lambda r: test_weekly(r["Sales Volume"], 1, 1), axis=1)

print("\n--- Synthetic Weekly Stats ---")
print("Base Weekly Describe:")
print(df["Weekly_Base"].describe())
print("\nBoosted Weekly Describe:")
print(df["Weekly_Boost"].describe())

print(f"\nPotential Max (Boosted): {df['Weekly_Boost'].max()}")
print(f"Count of products above 500 units/week: {len(df[df['Weekly_Boost'] > 500])}")
