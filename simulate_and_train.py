import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


def simulate_scenario(scenario: str, hours: int = 12, step_minutes: int = 1) -> pd.DataFrame:
    """Simulate sensor readings for a leakage scenario.

    Parameters
    ----------
    scenario: str
        One of "normal", "small", "medium", "catastrophic".
    hours: int
        Duration of the simulation in hours.
    step_minutes: int
        Time step in minutes.

    Returns
    -------
    pandas.DataFrame
        Dataframe of simulated sensor readings.
    """
    steps = int(hours * 60 / step_minutes)
    t = np.arange(steps) * step_minutes / 60.0

    # Base signals
    pressure = 5 + 0.1 * np.sin(2 * np.pi * t / 24)
    inflow = 100 + 5 * np.sin(2 * np.pi * t / 24)
    outflow = inflow.copy()
    temp = 20 + 5 * np.sin(2 * np.pi * t / 24)
    humidity = 50 + 10 * np.sin(2 * np.pi * t / 24)
    ph = 7 + 0.05 * np.sin(2 * np.pi * t / 24)
    level = 80 + 5 * np.sin(2 * np.pi * t / 24)

    leak_start = 4  # hours
    leak_idx = int(leak_start * 60 / step_minutes)
    leak_levels = {"normal": 0, "small": 5, "medium": 15, "catastrophic": 50}
    leak = leak_levels[scenario]

    outflow[leak_idx:] += leak
    pressure[leak_idx:] -= leak * 0.02
    level[leak_idx:] -= leak * np.cumsum(np.ones_like(level[leak_idx:]) * step_minutes) / 60.0

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=steps, freq=f"{step_minutes}min"),
            "pressure": pressure,
            "inflow": inflow,
            "outflow": outflow,
            "temperature": temp,
            "humidity": humidity,
            "pH": ph,
            "level": level,
        }
    )
    df["label_leak"] = 0
    if scenario != "normal":
        df.loc[leak_idx:, "label_leak"] = 1
    df["scenario"] = scenario
    return df


# 1) Generate data
scenarios = ["normal", "small", "medium", "catastrophic"]
dataset = pd.concat([simulate_scenario(s) for s in scenarios], ignore_index=True)
dataset.to_csv("synthetic_sensors.csv", index=False)


# 2) Create time windows
def create_windows(df: pd.DataFrame, window: int = 60):
    X, y = [], []
    features = ["pressure", "inflow", "outflow", "temperature", "humidity", "pH", "level"]
    data = df[features].values
    labels = df["label_leak"].values
    for i in range(len(df) - window):
        X.append(data[i : i + window])
        y.append(labels[i + window])
    return np.array(X), np.array(y)


X, y = create_windows(dataset, window=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


# 3) Build LSTM model
model = Sequential(
    [
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# 4) Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


# 5) Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
roc_auc = auc(fpr, tpr)

print("Confusion Matrix:\n", cm)
print("AUC:", roc_auc)


# 6) Example leak localization
def locate_leak(pressures: np.ndarray) -> int:
    """Locate leak based on largest pressure drop between sensors."""
    diff = np.diff(pressures)
    idx = np.argmin(diff)
    return idx


pressures_example = [5.0, 4.9, 4.1, 4.0]
segment_idx = locate_leak(np.array(pressures_example))
print(f"Leak suspected between sensor {segment_idx} and sensor {segment_idx + 1}")
