import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




df = pd.read_csv("C:/Users/susha/Downloads/multi_country_metro.csv")

print("Original Dataset Shape:", df.shape)


# 2. CLEAN DATA

# Clean Annual Ridership
df["Annual_Ridership_Million"] = (
    df["Annual_Ridership_Million"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0]
    .astype(float)
)

# Convert numeric columns
df["Lines"] = pd.to_numeric(df["Lines"], errors="coerce")

df["System_Length_km"] = pd.to_numeric(
    df["System_Length_km"],
    errors="coerce"
)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)


# 3. SELECT REQUIRED COLUMNS

model_df = df[
    [
        "Country",
        "City",
        "Metro",
        "Lines",
        "System_Length_km",
        "Year",
        "Annual_Ridership_Million"
    ]
].dropna()

print("After Cleaning:", model_df.shape)


# 4. CREATE METRO-LEVEL DATASET

# Dataset contains multiple stations for the same Metro.
# We keep one record per Metro to avoid station-level duplication.

metro_df = (
    model_df
    .sort_values("Year")
    .groupby("Metro", as_index=False)
    .last()
)

print("Unique Metro Systems:", metro_df.shape)


# 5. DEFINE FEATURES AND TARGET

X = metro_df[
    [
        "Country",
        "City",
        "Lines",
        "System_Length_km",
        "Year"
    ]
]

y = metro_df["Annual_Ridership_Million"]


# 6. DEFINE FEATURES

categorical_features = [
    "Country",
    "City"
]

numeric_features = [
    "Lines",
    "System_Length_km",
    "Year"
]


# 7. PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# 8. CREATE MODEL

model = ExtraTreesRegressor(
    n_estimators=300,
    random_state=42,
    max_features="sqrt",
    min_samples_leaf=2
)


# 9. CREATE PIPELINE

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 10. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# 11. TRAIN MODEL

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# 12. PREDICTION

y_pred = pipeline.predict(X_test)


# 13. MODEL EVALUATION

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("       MODEL PERFORMANCE")


print(f"MAE  : {mae:.2f} Million")
print(f"RMSE : {rmse:.2f} Million")
print(f"R²   : {r2:.4f}")



# 14. SAVE MODEL

model_filename = "metro_ridership_model.pkl"

joblib.dump(
    pipeline,
    model_filename
)

print(f"\nModel saved successfully as: {model_filename}")


# 15. TEST ONE SAMPLE PREDICTION

sample = pd.DataFrame({
    "Country": ["Japan"],
    "City": ["Tokyo"],
    "Lines": [13],
    "System_Length_km": [304],
    "Year": [2025]
})

prediction = pipeline.predict(sample)


print("       SAMPLE PREDICTION")


print(
    f"Predicted Annual Ridership: "
    f"{prediction[0]:.2f} Million"
)

