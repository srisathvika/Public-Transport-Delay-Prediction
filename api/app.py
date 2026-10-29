
from datetime import date as Date, datetime
from math import cos, pi, sin
import os

import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Public Transport Delay Prediction API",
    description=(
        "Predicts public transport arrival delay using "
        "the production model registered in MLflow."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# 2. MLflow configuration and model loading
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

REGISTERED_MODEL_NAME = "public-transport-delay-regressor"

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}@production"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the model assigned to the production alias.
# If loading fails, fix the MLflow connection/configuration
# rather than silently switching to another model.
try:
    MODEL_PATH = (
        "/app/notebooks/mlruns/1/models/"
        "m-921c1cf2419648ae84f5b68686f65401/artifacts"
        )
    model = mlflow.pyfunc.load_model(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Could not load production model: {MODEL_URI}. "
        f"Check that MLflow is running and the model "
        f"registry is accessible. Details: {exc}"
    ) from exc


# --------------------------------------------------
# 3. Input schema
# --------------------------------------------------

class TransportInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True
    )

    # Raw date and scheduled times
    date: Date
    scheduled_departure: str = Field(
        examples=["18:30:00"]
    )
    scheduled_arrival: str = Field(
        examples=["19:15:00"]
    )

    # Categorical features
    transport_type: str
    route_id: str
    origin_station: str
    destination_station: str
    weather_condition: str
    season: str

    # Leave event_type empty if the event type is unknown.
    event_type: str | None = None

    # Numeric contextual features
    temperature_C: float
    humidity_percent: float = Field(ge=0, le=100)
    wind_speed_kmh: float = Field(ge=0)
    precipitation_mm: float = Field(ge=0)
    traffic_congestion_index: float
    holiday: int = Field(ge=0, le=1)
    peak_hour: int = Field(ge=0, le=1)

    event_attendance_est: float = Field(
        default=0,
        ge=0,
    )


# --------------------------------------------------
# 4. Feature engineering
#    Reproduces Notebook 2
# --------------------------------------------------

def create_features(data: TransportInput) -> pd.DataFrame:
    """Convert raw API input into the 33 model input features."""

    try:
        departure = datetime.strptime(
            data.scheduled_departure,
            "%H:%M:%S",
        )
    except ValueError:
        try:
            departure = datetime.strptime(
                data.scheduled_departure,
                "%H:%M",
            )
        except ValueError as exc:
            raise ValueError(
                "scheduled_departure must use HH:MM or HH:MM:SS."
            ) from exc

    try:
        arrival = datetime.strptime(
            data.scheduled_arrival,
            "%H:%M:%S",
        )
    except ValueError:
        try:
            arrival = datetime.strptime(
                data.scheduled_arrival,
                "%H:%M",
            )
        except ValueError as exc:
            raise ValueError(
                "scheduled_arrival must use HH:MM or HH:MM:SS."
            ) from exc

    trip_date = pd.Timestamp(data.date)

    # Calendar features: same definitions as Notebook 2
    year = trip_date.year
    month = trip_date.month
    day_of_month = trip_date.day
    day_of_week = trip_date.day_name()
    weekday_num = trip_date.dayofweek
    is_weekend = int(weekday_num >= 5)
    quarter = trip_date.quarter

    # Scheduled time features
    departure_hour = departure.hour
    departure_minute = departure.minute
    arrival_hour = arrival.hour
    arrival_minute = arrival.minute

    # Minutes since midnight
    departure_minutes = (
        departure_hour * 60 + departure_minute
    )

    arrival_minutes = (
        arrival_hour * 60 + arrival_minute
    )

    # Duration, including trips crossing midnight
    scheduled_trip_duration_min = (
        arrival_minutes - departure_minutes
    ) % (24 * 60)

    # Cyclical time features
    departure_hour_sin = sin(
        2 * pi * departure_minutes / (24 * 60)
    )

    departure_hour_cos = cos(
        2 * pi * departure_minutes / (24 * 60)
    )

    # Event features
    attendance = data.event_attendance_est
    event_type = data.event_type or "Unknown / Missing Event"

    has_event = int(attendance > 0)

    # log1p matches Notebook 2's np.log1p
    log_event_attendance = float(np.log1p(attendance))

    # Construct exactly the features expected by the model.
    features = {
        "transport_type": data.transport_type,
        "route_id": data.route_id,
        "origin_station": data.origin_station,
        "destination_station": data.destination_station,
        "weather_condition": data.weather_condition,
        "event_type": event_type,
        "season": data.season,
        "day_of_week": day_of_week,

        "temperature_C": data.temperature_C,
        "humidity_percent": data.humidity_percent,
        "wind_speed_kmh": data.wind_speed_kmh,
        "precipitation_mm": data.precipitation_mm,
        "traffic_congestion_index": (
            data.traffic_congestion_index
        ),
        "holiday": data.holiday,
        "peak_hour": data.peak_hour,
        "event_attendance_est": attendance,
        "log_event_attendance": log_event_attendance,
        "has_event": has_event,
        "year": year,
        "month": month,
        "day_of_month": day_of_month,
        "weekday_num": weekday_num,
        "is_weekend": is_weekend,
        "quarter": quarter,
        "departure_hour": departure_hour,
        "departure_minute": departure_minute,
        "arrival_hour": arrival_hour,
        "arrival_minute": arrival_minute,
        "departure_minutes": departure_minutes,
        "arrival_minutes": arrival_minutes,
        "scheduled_trip_duration_min": (
            scheduled_trip_duration_min
        ),
        "departure_hour_sin": departure_hour_sin,
        "departure_hour_cos": departure_hour_cos,
    }

    # Match Notebook 2's feature order.
    feature_columns = [
        "transport_type",
        "route_id",
        "origin_station",
        "destination_station",
        "weather_condition",
        "event_type",
        "season",
        "day_of_week",
        "temperature_C",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "traffic_congestion_index",
        "holiday",
        "peak_hour",
        "event_attendance_est",
        "log_event_attendance",
        "has_event",
        "year",
        "month",
        "day_of_month",
        "weekday_num",
        "is_weekend",
        "quarter",
        "departure_hour",
        "departure_minute",
        "arrival_hour",
        "arrival_minute",
        "departure_minutes",
        "arrival_minutes",
        "scheduled_trip_duration_min",
        "departure_hour_sin",
        "departure_hour_cos",
    ]

    return pd.DataFrame(
        [features],
        columns=feature_columns,
    )


# --------------------------------------------------
# 5. Health endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Public Transport Delay Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_URI,
    }


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: TransportInput):
    try:
        # Transform raw inputs into the model's 33 features.
        features = create_features(data)
        
        
        # Match the input data types expected by the registered MLflow model
        integer_columns = [
            "humidity_percent",
            "wind_speed_kmh",
            "traffic_congestion_index",
            "holiday",
            "peak_hour",
            "event_attendance_est",
            "has_event",
            "year",
            "month",
            "day_of_month",
            "weekday_num",
            "is_weekend",
            "quarter",
            "departure_hour",
            "departure_minute",
            "arrival_hour",
            "arrival_minute",
            "departure_minutes",
            "arrival_minutes",
            "scheduled_trip_duration_min",
        ]

        float_columns = [
            "temperature_C",
            "precipitation_mm",
            "log_event_attendance",
            "departure_hour_sin",
            "departure_hour_cos",
        ]

        # Reject fractional values for columns the model expects as integers
        for column in integer_columns:
            if column in features.columns:
                values = pd.to_numeric(features[column], errors="raise")
                if values.isna().any() or (values % 1 != 0).any():
                    raise ValueError(
                        f"{column} must contain whole-number values."
                    )
                features[column] = values.astype("int64")

        for column in float_columns:
            if column in features.columns:
                features[column] = pd.to_numeric(
                    features[column], errors="raise"
                ).astype("float64")

        prediction = model.predict(features)


                # Predict using the MLflow production model.

        predicted_delay = float(prediction[0])

        return {
            "predicted_delay_minutes": round(
                predicted_delay, 2
            ),
            "predicted_delayed": bool(
                predicted_delay > 5
            ),
            "delay_threshold_minutes": 5,
            "model": MODEL_URI,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs."
        ) from e
