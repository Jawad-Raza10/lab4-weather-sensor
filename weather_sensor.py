import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

# Weather sensor data
data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

# Keep original temperature values
df["Temp_raw"] = df["Temperature"]

# --------------------------------------------------
# 1. Statistics BEFORE cleaning
# --------------------------------------------------

print("Stats BEFORE cleaning:")

print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())

before_skewness = stats.skew(df["Temperature"].dropna())
before_kurtosis = stats.kurtosis(df["Temperature"].dropna())

print("Skewness:", before_skewness)
print("Kurtosis:", before_kurtosis)

# --------------------------------------------------
# 2. Fill missing values using linear interpolation
# --------------------------------------------------

df["Temperature"] = df["Temperature"].interpolate(method="linear")

print("\nAfter interpolation, missing values:",
      df["Temperature"].isnull().sum())

# --------------------------------------------------
# 3. Detect outlier using Z-score threshold = 2
# --------------------------------------------------

z_scores = zscore(df["Temperature"])

df["Temp_Zscore_Before_Cleaning"] = z_scores

outlier_mask = df["Temp_Zscore_Before_Cleaning"].abs() > 2

outliers = df[outlier_mask]

print("\nZ-score(>2) outlier detected:")

for index, row in outliers.iterrows():
    print(
        "Hour:", row["Hour"],
        "Value:", row["Temperature"]
    )

# --------------------------------------------------
# 4. Replace outlier with median
# --------------------------------------------------

median_temperature = df["Temperature"].median()

df.loc[outlier_mask, "Temperature"] = median_temperature

print("\nReplaced outlier with column median:",
      median_temperature)

# --------------------------------------------------
# 5. Final Z-score standardization
# --------------------------------------------------

df["Temp_Zscore"] = zscore(df["Temperature"])

# --------------------------------------------------
# 6. Before/After comparison
# --------------------------------------------------

comparison = df[
    ["Hour", "Temp_raw", "Temperature", "Temp_Zscore"]
]

comparison = comparison.rename(
    columns={"Temperature": "Temp_clean"}
)

print("\nBefore/After comparison:")
print(comparison)

# --------------------------------------------------
# 7. Statistics AFTER cleaning
# --------------------------------------------------

print("\nStats AFTER cleaning:")

print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())

print("\nMean BEFORE cleaning:",
      df["Temp_raw"].mean())

print("Std BEFORE cleaning:",
      df["Temp_raw"].std())

print("\nMean AFTER cleaning:",
      df["Temperature"].mean())

print("Std AFTER cleaning:",
      df["Temperature"].std())