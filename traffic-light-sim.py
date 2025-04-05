import streamlit as st
import time

# ----- Configuration for Simulation -----
# Define durations (in seconds) for each traffic light phase:
RED_DURATION = 30      # Red light lasts 30 seconds
YELLOW_DURATION = 5    # Yellow light lasts 5 seconds
GREEN_DURATION = 25    # Green light lasts 25 seconds
CYCLE_DURATION = RED_DURATION + YELLOW_DURATION + GREEN_DURATION

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
      - remaining time in that phase,
      - and the distance to the intersection (in meters).
    
    This is a very simple heuristic:
    - If the light is Green, assume a default cruising speed.
    - If the light is Yellow or Red, compute a speed that would allow the driver to reach the light as the phase changes.
    
    Note: 1 m/s = 3.6 km/h.
    """
    if distance <= 0:
        return 0
    # Convert remaining time to hours
    time_hours = remaining_time / 3600.0
    # Calculate the speed (m/s) needed to cover the distance in the remaining time:
    # If remaining_time is too short, this value may be very high.
    speed_mps = distance / remaining_time if remaining_time > 0 else 0
    speed_kmh = speed_mps * 3.6

    if phase == "Green":
        # If green, use a default safe speed (or you might choose a different approach)
        return 50  # e.g., 50 km/h
    else:
        # For Red or Yellow, suggest the computed speed (but cap it to a realistic maximum)
        return min(speed_kmh, 80)

# ----- Streamlit UI -----
st.set_page_config(page_title="Traffic Light Simulation & Speed Suggestion", layout="centered")
st.title("🚦 Traffic Light Phase Simulator & Speed Advisor")
st.write("This simulation displays a traffic light's current phase, a countdown timer, and suggests a driving speed based on your distance to the intersection.")

# Get current time in the cycle (using system time modulo cycle duration)
current_cycle_time = time.time() % CYCLE_DURATION
phase, time_remaining = get_traffic_light_phase(current_cycle_time)

# Display traffic light phase and countdown timer
st.markdown(f"### Current Traffic Light Phase: **{phase}**")
st.markdown(f"#### Time remaining in this phase: **{int(time_remaining)} seconds**")

# Input: Distance to intersection (in meters)
distance = st.number_input("Enter distance to the traffic light (in meters):", min_value=0.0, value=100.0)

# Suggest a speed based on current phase, remaining time, and distance
recommended_speed = suggest_speed(phase, time_remaining, distance)
st.markdown(f"### Recommended Speed: **{recommended_speed:.2f} km/h**")

# Optional: Refresh the simulation periodically
refresh_interval = st.slider("Auto-refresh interval (seconds)", min_value=1, max_value=10, value=3)
st.write("The simulation auto-refreshes based on the interval above.")

# Automatically refresh the page to update the simulation (this uses Streamlit's experimental feature)
st.experimental_rerun()  # Note: In production, you might prefer st_autorefresh
