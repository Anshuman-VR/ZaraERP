import pandas as pd
import numpy as np
import os

BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP"
DATA_PATH = os.path.join(BASE, "data", "zaraSales.csv")

df = pd.read_csv(DATA_PATH, sep=';')
df["Promotion"] = pd.to_numeric(df["Promotion"], errors="coerce").fillna(0)
df["Seasonal"] = pd.to_numeric(df["Seasonal"], errors="coerce").fillna(0)

# Weekly target (6-week baseline)
def estimate_shelf_life(row):
    base_weeks = 6.0
    if row["Seasonal"] == 1: base_weeks = 3.0
    if row["Promotion"] == 1: base_weeks *= 0.5
    return max(1.0, base_weeks)

df["Shelf_Life_Weeks"] = df.apply(estimate_shelf_life, axis=1)
df["Sales_Volume_Weekly"] = df["Sales Volume"] / df["Shelf_Life_Weeks"]

print("--- Value Counts ---")
print(df["Promotion"].value_counts())
print(df["Seasonal"].value_counts())

print("\n--- Correlation with Target ---")
print(df[["Promotion", "Seasonal", "Sales_Volume_Weekly"]].corr())

print("\n--- Grouped Means ---")
print(df.groupby("Promotion")["Sales_Volume_Weekly"].mean())
print(df.groupby("Seasonal")["Sales_Volume_Weekly"].mean())
