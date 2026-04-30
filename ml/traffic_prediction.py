"""
ML Traffic Prediction Module

This module trains a simple scikit-learn model to predict road congestion
levels from temporal Cairo traffic data.

The predicted congestion level can later be used inside Dijkstra by replacing
static edge weights with traffic-aware predicted weights.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Note:
# Project explanations and analysis are documented in the report/README,
# not inside the code.
MODEL_FILE = os.path.join(os.path.dirname(__file__), "traffic_congestion_model.pkl")
CATEGORICAL_COLUMNS = ["road_id", "time_period"]
NUMERIC_COLUMNS = ["distance_km", "capacity", "condition"]


def calculate_congestion_ratio(traffic_flow, capacity):
    """Calculate congestion ratio using traffic flow and road capacity."""
    if capacity == 0:
        return 0
    return traffic_flow / capacity


def get_congestion_level(congestion_ratio):
    """Convert a congestion ratio into Low, Medium, or High."""
    if congestion_ratio < 0.5:
        return "Low"
    if congestion_ratio < 0.8:
        return "Medium"
    return "High"


def create_encoder():
    """Create a OneHotEncoder that works with old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_project_road_metadata():
    """Return the real road metadata provided for the Cairo project."""
    return {
        "1-3": {"distance_km": 8.5, "capacity": 3000, "condition": 7},
        "1-8": {"distance_km": 6.2, "capacity": 2500, "condition": 6},
        "2-3": {"distance_km": 5.9, "capacity": 2800, "condition": 8},
        "2-5": {"distance_km": 4.0, "capacity": 3200, "condition": 9},
        "3-5": {"distance_km": 6.1, "capacity": 3500, "condition": 7},
        "3-6": {"distance_km": 3.2, "capacity": 2000, "condition": 8},
        "3-9": {"distance_km": 4.5, "capacity": 2600, "condition": 6},
        "3-10": {"distance_km": 3.8, "capacity": 2400, "condition": 7},
        "4-2": {"distance_km": 15.2, "capacity": 3800, "condition": 9},
        "4-14": {"distance_km": 5.3, "capacity": 3000, "condition": 10},
        "5-11": {"distance_km": 7.9, "capacity": 3100, "condition": 7},
        "6-9": {"distance_km": 2.2, "capacity": 1800, "condition": 8},
        "7-8": {"distance_km": 24.5, "capacity": 3500, "condition": 8},
        "7-15": {"distance_km": 9.8, "capacity": 3000, "condition": 9},
        "8-10": {"distance_km": 3.3, "capacity": 2200, "condition": 7},
        "8-12": {"distance_km": 14.8, "capacity": 2600, "condition": 5},
        "9-10": {"distance_km": 2.1, "capacity": 1900, "condition": 7},
        "10-11": {"distance_km": 8.7, "capacity": 2400, "condition": 6},
        "11-F2": {"distance_km": 3.6, "capacity": 2200, "condition": 7},
        "12-1": {"distance_km": 12.7, "capacity": 2800, "condition": 6},
        "13-4": {"distance_km": 45.0, "capacity": 4000, "condition": 10},
        "14-13": {"distance_km": 35.5, "capacity": 3800, "condition": 9},
        "15-7": {"distance_km": 9.8, "capacity": 3000, "condition": 9},
        "F1-5": {"distance_km": 7.5, "capacity": 3500, "condition": 9},
        "F1-2": {"distance_km": 9.2, "capacity": 3200, "condition": 8},
        "F2-3": {"distance_km": 2.5, "capacity": 2000, "condition": 7},
        "F7-15": {"distance_km": 8.3, "capacity": 2800, "condition": 8},
        "F8-4": {"distance_km": 6.1, "capacity": 3000, "condition": 9},
    }


def get_project_traffic_data():
    """Return the real temporal traffic data provided for the Cairo project."""
    return {
        "1-3": {
            "Morning Peak": 2800,
            "Afternoon": 1500,
            "Evening Peak": 2600,
            "Night": 800,
        },
        "1-8": {
            "Morning Peak": 2200,
            "Afternoon": 1200,
            "Evening Peak": 2100,
            "Night": 600,
        },
        "2-3": {
            "Morning Peak": 2700,
            "Afternoon": 1400,
            "Evening Peak": 2500,
            "Night": 700,
        },
        "2-5": {
            "Morning Peak": 3000,
            "Afternoon": 1600,
            "Evening Peak": 2800,
            "Night": 650,
        },
        "3-5": {
            "Morning Peak": 3200,
            "Afternoon": 1700,
            "Evening Peak": 3100,
            "Night": 800,
        },
        "3-6": {
            "Morning Peak": 1800,
            "Afternoon": 1400,
            "Evening Peak": 1900,
            "Night": 500,
        },
        "3-9": {
            "Morning Peak": 2400,
            "Afternoon": 1300,
            "Evening Peak": 2200,
            "Night": 550,
        },
        "3-10": {
            "Morning Peak": 2300,
            "Afternoon": 1200,
            "Evening Peak": 2100,
            "Night": 500,
        },
        "4-2": {
            "Morning Peak": 3600,
            "Afternoon": 1800,
            "Evening Peak": 3300,
            "Night": 750,
        },
        "4-14": {
            "Morning Peak": 2800,
            "Afternoon": 1600,
            "Evening Peak": 2600,
            "Night": 600,
        },
        "5-11": {
            "Morning Peak": 2900,
            "Afternoon": 1500,
            "Evening Peak": 2700,
            "Night": 650,
        },
        "6-9": {
            "Morning Peak": 1700,
            "Afternoon": 1300,
            "Evening Peak": 1800,
            "Night": 450,
        },
        "7-8": {
            "Morning Peak": 3200,
            "Afternoon": 1700,
            "Evening Peak": 3000,
            "Night": 700,
        },
        "7-15": {
            "Morning Peak": 2800,
            "Afternoon": 1500,
            "Evening Peak": 2600,
            "Night": 600,
        },
        "8-10": {
            "Morning Peak": 2000,
            "Afternoon": 1100,
            "Evening Peak": 1900,
            "Night": 450,
        },
        "8-12": {
            "Morning Peak": 2400,
            "Afternoon": 1300,
            "Evening Peak": 2200,
            "Night": 500,
        },
        "9-10": {
            "Morning Peak": 1800,
            "Afternoon": 1200,
            "Evening Peak": 1700,
            "Night": 400,
        },
        "10-11": {
            "Morning Peak": 2200,
            "Afternoon": 1300,
            "Evening Peak": 2100,
            "Night": 500,
        },
        "11-F2": {
            "Morning Peak": 2100,
            "Afternoon": 1200,
            "Evening Peak": 2000,
            "Night": 450,
        },
        "12-1": {
            "Morning Peak": 2600,
            "Afternoon": 1400,
            "Evening Peak": 2400,
            "Night": 550,
        },
        "13-4": {
            "Morning Peak": 3800,
            "Afternoon": 2000,
            "Evening Peak": 3500,
            "Night": 800,
        },
        "14-13": {
            "Morning Peak": 3600,
            "Afternoon": 1900,
            "Evening Peak": 3300,
            "Night": 750,
        },
        "15-7": {
            "Morning Peak": 2800,
            "Afternoon": 1500,
            "Evening Peak": 2600,
            "Night": 600,
        },
        "F1-5": {
            "Morning Peak": 3300,
            "Afternoon": 2200,
            "Evening Peak": 3100,
            "Night": 1200,
        },
        "F1-2": {
            "Morning Peak": 3000,
            "Afternoon": 2000,
            "Evening Peak": 2800,
            "Night": 1100,
        },
        "F2-3": {
            "Morning Peak": 1900,
            "Afternoon": 1600,
            "Evening Peak": 1800,
            "Night": 900,
        },
        "F7-15": {
            "Morning Peak": 2600,
            "Afternoon": 1500,
            "Evening Peak": 2400,
            "Night": 550,
        },
        "F8-4": {
            "Morning Peak": 2800,
            "Afternoon": 1600,
            "Evening Peak": 2600,
            "Night": 600,
        },
    }


def create_traffic_dataset(road_metadata, traffic_data):
    """
    Create a training dataset from road metadata and temporal traffic flow data.

    road_metadata format:
    {
        "1-3": {"distance_km": 4.2, "capacity": 3000, "condition": 8}
    }

    traffic_data format:
    {
        "1-3": {
            "Morning Peak": 2800,
            "Afternoon": 1500,
            "Evening Peak": 2600,
            "Night": 800
        }
    }
    """
    rows = []

    for road_id, time_values in traffic_data.items():
        if road_id not in road_metadata:
            print(f"Warning: road metadata missing for {road_id}, skipping.")
            continue

        road_info = road_metadata[road_id]

        for time_period, traffic_flow in time_values.items():
            congestion_ratio = calculate_congestion_ratio(
                traffic_flow, road_info["capacity"]
            )

            rows.append(
                {
                    "road_id": road_id,
                    "distance_km": road_info["distance_km"],
                    "capacity": road_info["capacity"],
                    "condition": road_info["condition"],
                    "time_period": time_period,
                    "traffic_flow": traffic_flow,
                    "congestion_ratio": congestion_ratio,
                    "congestion_level": get_congestion_level(congestion_ratio),
                }
            )

    return pd.DataFrame(rows)


def prepare_features(dataframe, encoder=None, training=True):
    """Encode categorical columns and combine them with numeric columns."""
    numeric_features = dataframe[NUMERIC_COLUMNS].reset_index(drop=True)

    if training:
        encoder = create_encoder()
        encoded_values = encoder.fit_transform(dataframe[CATEGORICAL_COLUMNS])
    else:
        encoded_values = encoder.transform(dataframe[CATEGORICAL_COLUMNS])

    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
    encoded_dataframe = pd.DataFrame(encoded_values, columns=encoded_columns)

    features = pd.concat([numeric_features, encoded_dataframe], axis=1)
    return features, encoder


def train_traffic_model(dataframe, model_path=MODEL_FILE):
    """Train, evaluate, and save the traffic congestion prediction model."""
    features, encoder = prepare_features(dataframe, training=True)
    target = dataframe["congestion_level"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("Traffic Congestion Model Evaluation")
    print("-" * 60)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    model_bundle = {
        "model": model,
        "encoder": encoder,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # The .pkl file stores the trained model in binary format and is loaded later
    # using joblib.load(). It is not meant to be opened manually.
    joblib.dump(model_bundle, model_path)
    print(f"\nModel saved to: {model_path}")

    return model_bundle


def predict_congestion(road_id, distance_km, capacity, condition, time_period):
    """Load the trained model and predict Low, Medium, or High congestion."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Model file not found. Please run train_traffic_model() first to "
            "create traffic_congestion_model.pkl."
        )

    model_bundle = joblib.load(MODEL_FILE)
    model = model_bundle["model"]
    encoder = model_bundle["encoder"]

    # Unknown road IDs or time periods can reduce prediction accuracy because
    # the model has not learned patterns for them from the training data.
    if road_id not in get_project_road_metadata():
        print(f"Warning: Unknown road_id {road_id}, prediction may be inaccurate.")

    allowed_time_periods = ["Morning Peak", "Afternoon", "Evening Peak", "Night"]
    if time_period not in allowed_time_periods:
        print(
            "Warning: Unknown time_period "
            f"{time_period}, expected one of [Morning Peak, Afternoon, "
            "Evening Peak, Night]"
        )

    # OneHotEncoder(handle_unknown="ignore") lets prediction continue safely
    # even when a category was not seen during training.
    input_data = pd.DataFrame(
        [
            {
                "road_id": road_id,
                "distance_km": distance_km,
                "capacity": capacity,
                "condition": condition,
                "time_period": time_period,
            }
        ]
    )

    features, _ = prepare_features(input_data, encoder=encoder, training=False)
    prediction = model.predict(features)[0]
    return prediction


def get_predicted_weight(base_distance, predicted_congestion_level):
    """
    Convert predicted congestion into a routing weight.

    This predicted weight can be used inside Dijkstra instead of static edge
    weights, so routes avoid roads that are expected to be congested.
    """
    if predicted_congestion_level == "Low":
        return base_distance * 1.0
    if predicted_congestion_level == "Medium":
        return base_distance * 1.5
    return base_distance * 2.0


def plot_congestion_distribution(dataframe, block=True):
    """Show a bar chart of Low, Medium, and High congestion cases."""
    order = ["Low", "Medium", "High"]
    counts = dataframe["congestion_level"].value_counts().reindex(order, fill_value=0)

    counts.plot(kind="bar", color=["green", "orange", "red"])
    plt.title("Congestion Level Distribution")
    plt.xlabel("Congestion Level")
    plt.ylabel("Number of Cases")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show(block=block)

    if not block:
        plt.pause(2)
        plt.close()


if __name__ == "__main__":
    road_metadata = get_project_road_metadata()
    traffic_data = get_project_traffic_data()

    traffic_dataframe = create_traffic_dataset(road_metadata, traffic_data)
    print("Project traffic dataset preview:")
    print(traffic_dataframe.head())
    print()

    train_traffic_model(traffic_dataframe)

    predicted_level = predict_congestion(
        road_id="1-3",
        distance_km=8.5,
        capacity=3000,
        condition=7,
        time_period="Morning Peak",
    )
    predicted_weight = get_predicted_weight(8.5, predicted_level)

    print("\nSample prediction:")
    print(f"Road 1-3 during Morning Peak: {predicted_level}")
    print(f"Predicted Dijkstra weight: {predicted_weight}")

    plot_congestion_distribution(traffic_dataframe, block=False)
