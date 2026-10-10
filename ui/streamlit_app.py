
import requests
import streamlit as st
from datetime import date, time

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Transport Delay Predictor",
    page_icon="🚌",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------
# Categories from the dataset
# --------------------------------------------------
TRANSPORT_OPTIONS = ["Bus", "Metro", "Train", "Tram"]

WEATHER_OPTIONS = [
    "Clear",
    "Cloudy",
    "Fog",
    "Rain",
    "Snow",
    "Storm",
]

ROUTE_OPTIONS = [f"Route_{i}" for i in range(1, 21)]

STATION_OPTIONS = [f"Station_{i}" for i in range(1, 51)]

SEASON_OPTIONS = ["Winter", "Spring", "Summer", "Autumn"]

EVENT_OPTIONS = [
    "Unknown / Missing Event",
    "Festival",
    "Sports",
    "Concert",
    "Parade",
    "Protest",
]

# --------------------------------------------------
# Page heading
# --------------------------------------------------
st.title("Public Transport Delay Predictor")

st.info(
    "Enter the planned journey, weather conditions and event "
    "information to estimate the arrival delay."
)

# --------------------------------------------------
# Prediction form
# --------------------------------------------------
with st.form("prediction_form"):

    # 1. Journey details
    st.subheader("1. Journey Details")

    col1, col2 = st.columns(2)

    with col1:
        travel_date = st.date_input(
            "Travel date",
            value=date.today(),
        )

        scheduled_departure = st.time_input(
            "Scheduled departure",
            value=time(18, 30),
        )

        transport_type = st.selectbox(
            "Transport type",
            options=TRANSPORT_OPTIONS,
        )

        route_id = st.selectbox(
            "Route ID",
            options=ROUTE_OPTIONS,
        )

        origin_station = st.selectbox(
            "Origin station",
            options=STATION_OPTIONS,
        )

    with col2:
        scheduled_arrival = st.time_input(
            "Scheduled arrival",
            value=time(19, 15),
        )

        destination_station = st.selectbox(
            "Destination station",
            options=STATION_OPTIONS,
            index=1,
        )

        weather_condition = st.selectbox(
            "Weather condition",
            options=WEATHER_OPTIONS,
        )

        season = st.selectbox(
            "Season",
            options=SEASON_OPTIONS,
        )

        event_type = st.selectbox(
            "Event type",
            options=EVENT_OPTIONS,
        )

    st.divider()

    # 2. Weather measurements
    st.subheader("2. Weather Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-30.0,
            max_value=55.0,
            value=15.0,
            step=0.5,
        )

    with col2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=70,
        )

    with col3:
        wind_speed = st.number_input(
            "Wind speed (km/h)",
            min_value=0,
            value=10,
        )

    col1, col2 = st.columns(2)

    with col1:
        precipitation = st.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            value=0.0,
            step=0.5,
        )

    with col2:
        traffic = st.number_input(
            "Traffic congestion index",
            min_value=0,
            value=5,
        )

    st.divider()

    # 3. Other conditions
    st.subheader("3. Other Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        holiday = st.selectbox(
            "Holiday",
            options=[0, 1],
            format_func=lambda value: (
                "Yes" if value == 1 else "No"
            ),
        )

    with col2:
        peak_hour = st.selectbox(
            "Peak hour",
            options=[0, 1],
            format_func=lambda value: (
                "Yes" if value == 1 else "No"
            ),
        )

    with col3:
        event_attendance = st.number_input(
            "Estimated event attendance",
            min_value=0,
            value=0,
            step=100,
        )

    submitted = st.form_submit_button(
        "Predict Arrival Delay",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Validate input and call FastAPI
# --------------------------------------------------
if submitted:

    if origin_station == destination_station:
        st.error(
            "Origin and destination stations must be different."
        )

    elif scheduled_arrival <= scheduled_departure:
        st.error(
            "Scheduled arrival must be later than departure. "
            "Overnight journeys are not supported in this version."
        )

    else:

        payload = {
            "date": travel_date.isoformat(),
            "scheduled_departure": scheduled_departure.strftime(
                "%H:%M:%S"
            ),
            "scheduled_arrival": scheduled_arrival.strftime(
                "%H:%M:%S"
            ),
            "transport_type": transport_type,
            "route_id": route_id,
            "origin_station": origin_station,
            "destination_station": destination_station,
            "weather_condition": weather_condition,
            "season": season,
            "event_type": event_type,
            "temperature_C": temperature,
            "humidity_percent": humidity,
            "wind_speed_kmh": wind_speed,
            "precipitation_mm": precipitation,
            "traffic_congestion_index": traffic,
            "holiday": holiday,
            "peak_hour": peak_hour,
            "event_attendance_est": event_attendance,
        }

        try:
            with st.spinner("Generating prediction..."):
                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=30,
                )

            if response.ok:
                result = response.json()

                st.divider()
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Predicted arrival delay",
                        value=(
                            f"{result['predicted_delay_minutes']:.2f} "
                            "minutes"
                        ),
                    )

                with col2:
                    is_delayed = result["predicted_delayed"]

                    st.metric(
                        label="Predicted trip status",
                        value=(
                            "Delayed"
                            if is_delayed
                            else "Not delayed"
                        ),
                    )

                if is_delayed:
                    st.warning(
                        "The model predicts an arrival delay "
                        "greater than the configured threshold."
                    )
                else:
                    st.success(
                        "The model predicts an arrival delay "
                        "within the configured threshold."
                    )

                st.caption(
                    "Classification threshold: "
                    f"{result['delay_threshold_minutes']} minutes"
                )

                st.caption(
                    "Predictions are estimates and are not "
                    "guarantees of actual journey performance."
                )

            else:
                st.error(
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Ensure the API is "
                "running at http://127.0.0.1:8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. Check that "
                "FastAPI and MLflow are running."
            )

        except (ValueError, KeyError) as exc:
            st.error(
                f"Could not process the API response: {exc}"
            )
