import pandas as pd
import numpy as np

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
           "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", 
           "length", "width", "height", "curb-weight", "engine-type", 
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", 
           "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.read_csv(filename, names=headers)

df.replace("?", np.nan, inplace=True)

avg_stroke = df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)
print(f"Середнє значення stroke: {avg_stroke}")

df["height"] = df["height"] / df["height"].max()

dummy_aspiration = pd.get_dummies(df["aspiration"])

df = pd.concat([df, dummy_aspiration], axis=1)
df.drop("aspiration", axis=1, inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)

print(df.head())
df.to_csv("clean_auto.csv", index=False)