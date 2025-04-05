import streamlit as st
import time
from streamlit_autorefresh import st_autorefresh
import folium
from streamlit_folium import st_folium

# ---------------- Configuration -----------------
# Traffic light simulation durations (in seconds)
RED_DURATION = 30
YELLOW_DURATION = 5
GREEN_DURATION = 25
CYCLE_DURATION = RED_DURATION + YELLOW_DURATION + GREEN_DURATION

# Dummy geographic data for Chandigarh city
# (These coordinates can be replaced with real intersection coordinates)
CHD_LAT = 30.7333  
CHD_LON = 76.7794  

# ---------------- Helper Functions -----------------
def get_traffic_light_phase(current_time):
    """Determine the current phase and the remaining time for that phase."""
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
      - Current phase,
      - Remaining time in the current phase,
      - Simulated distance to the intersection (in meters).
    """
    if distance <= 0 or remaining_time <= 0:
        return 0
    # For green light, use a default cruising speed.
    if phase == "Green":
        return 50
    else:
        # Calculate required speed (m/s) to cover the distance in the remaining time
        speed_mps = distance / remaining_time  
        # Convert m/s to km/h (1 m/s = 3.6 km/h)
        speed_kmh = speed_mps * 3.6
        return min(speed_kmh, 80)  # Cap at 80 km/h for safety

# ---------------- Auto-Refresh -----------------
# Refresh the app every second (1000 ms)
st_autorefresh(interval=1000, limit=None, key="auto-refresh")

# ---------------- Simulation Logic -----------------
# Compute current cycle time (in seconds) using system time modulo cycle duration
current_cycle_time = time.time() % CYCLE_DURATION
phase, time_remaining = get_traffic_light_phase(current_cycle_time)

# ----------------- UI -----------------
st.set_page_config(page_title="Traffic Light Simulator & Speed Advisor", layout="centered")
st.title("🚦 Traffic Light Simulation & Speed Suggestion")

# Display current traffic light phase and remaining time
st.markdown(f"### Current Traffic Light Phase: **{phase}**")
st.markdown(f"#### Time Remaining in Phase: **{int(time_remaining)} seconds**")

# Dummy distance input for Chandigarh (in meters)
distance = st.number_input("Enter distance to the traffic light (in meters):", min_value=0.0, value=100.0)

# Calculate recommended speed based on current phase, remaining time, and distance
recommended_speed = suggest_speed(phase, time_remaining, distance)
st.markdown(f"### Recommended Speed: **{recommended_speed:.2f} km/h**")

# ---------------- Map Integration -----------------
st.markdown("### Intersection Location in Chandigarh")
# Create a folium map centered at the dummy intersection in Chandigarh
m = folium.Map(location=[CHD_LAT, CHD_LON], zoom_start=15)
folium.Marker([CHD_LAT, CHD_LON], popup="Intersection").add_to(m)

# Display the map in the app
st_folium(m, width=700, height=500)
