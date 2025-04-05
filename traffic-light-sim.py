import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(page_title="Traffic Light Simulator & Speed Advisor", layout="centered")

import time
from streamlit_autorefresh import st_autorefresh
import folium
from streamlit_folium import st_folium

# ----- Configuration for Simulation -----
# Define durations (in seconds) for each traffic light phase:
RED_DURATION = 30      # Red light lasts 30 seconds
YELLOW_DURATION = 5    # Yellow light lasts 5 seconds
GREEN_DURATION = 25    # Green light lasts 25 seconds
CYCLE_DURATION = RED_DURATION + YELLOW_DURATION + GREEN_DURATION

# Dummy geographic data for Chandigarh city (adjust as needed)
CHD_LAT = 30.7333  
CHD_LON = 76.7794  

# ----- Helper Functions -----
def get_traffic_light_phase(current_time):
    """
    Given the current time (in seconds modulo the cycle duration),
    determine the current phase (Red, Yellow, Green) and time remaining in that phase.
    """
    if current_time < RED_DURATION:
        phase = "Red"
        remaining = RED_DURATION - current_time
    elif current_time < RED_DURATION + YELLOW_DURATION:
        phase = "Yellow"
        remaining = RED_DURATION + YELLOW_DURATION - current_time
    else:
        phase = "Green"
        remaining = CYCLE_DURATION - current_time
    return phase, remaining

def suggest_speed(phase, remaining_time, distance):
    """
    Suggest a speed (in km/h) based on:
      - the current traffic light phase,
      - remaining time in the current phase,
      - and the distance to the intersection (in meters).
    
    This is a very simple heuristic:
    - For green, a default safe speed is used.
    - For red or yellow, we compute the speed needed to cover the distance in the remaining time.
    Note: 1 m/s = 3.6 km/h.
    """
    if distance <= 0 or remaining_time <= 0:
        return 0
    if phase == "Green":
        return 50  # e.g., 50 km/h default cruising speed for green
    else:
        speed_mps = distance / remaining_time
        speed_kmh = speed_mps * 3.6
        return min(speed_kmh, 80)  # Cap speed at 80 km/h

# ----- Auto-Refresh Setup -----
# Refresh the app every 1 second (1000 ms)
st_autorefresh(interval=1000, limit=None, key="auto-refresh")

# ----- Simulation Logic -----
# Compute current cycle time (in seconds) using system time modulo the cycle duration
current_cycle_time = time.time() % CYCLE_DURATION
phase, time_remaining = get_traffic_light_phase(current_cycle_time)

# ----- Streamlit UI -----
st.title("🚦 Traffic Light Simulation & Speed Advisor")
st.write("This simulation displays the current traffic light phase, a countdown timer, and suggests a driving speed based on your distance to the light.")

# Display current phase and remaining time
st.markdown(f"### Current Traffic Light Phase: **{phase}**")
st.markdown(f"#### Time Remaining in Phase: **{int(time_remaining)} seconds**")

# Input for distance to the traffic light (in meters)
distance = st.number_input("Enter distance to the traffic light (in meters):", min_value=0.0, value=100.0)

# Calculate recommended speed
recommended_speed = suggest_speed(phase, time_remaining, distance)
st.markdown(f"### Recommended Speed: **{recommended_speed:.2f} km/h**")

# ----- Map Integration with Folium -----
st.markdown("### Intersection Location in Chandigarh")
m = folium.Map(location=[CHD_LAT, CHD_LON], zoom_start=15)
folium.Marker([CHD_LAT, CHD_LON], popup="Intersection").add_to(m)
st_folium(m, width=700, height=500)
