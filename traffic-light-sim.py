import streamlit as st
import time

# ----- Configuration for Simulation -----
RED_DURATION = 30      # Red light duration (seconds)
YELLOW_DURATION = 5    # Yellow light duration (seconds)
GREEN_DURATION = 25    # Green light duration (seconds)
CYCLE_DURATION = RED_DURATION + YELLOW_DURATION + GREEN_DURATION

# ----- Helper Functions -----
def get_traffic_light_phase(current_time):
    """
    Determine the traffic light phase and remaining time in that phase.
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
    Suggest a speed (in km/h) based on current phase, remaining time, and distance.
    This is a simple heuristic.
    """
    if distance <= 0:
        return 0
    # For green, suggest a default speed; for others, compute speed needed.
    if phase == "Green":
        return 50  # default cruising speed (km/h)
    else:
        speed_mps = distance / remaining_time if remaining_time > 0 else 0
        speed_kmh = speed_mps * 3.6
        return min(speed_kmh, 80)  # cap the speed at 80 km/h

# ----- Streamlit UI -----
st.set_page_config(page_title="Traffic Light Simulation & Speed Suggestion", layout="centered")
st.title("🚦 Traffic Light Phase Simulator & Speed Advisor")
st.write("This simulation displays the current traffic light phase, a countdown timer, and suggests a driving speed based on your distance to the light.")

# Get current cycle time using system time modulo the cycle duration
current_cycle_time = time.time() % CYCLE_DURATION
phase, time_remaining = get_traffic_light_phase(current_cycle_time)

# Display the current phase and time remaining
st.markdown(f"### Current Phase: **{phase}**")
st.markdown(f"#### Time remaining: **{int(time_remaining)} seconds**")

# Input for the distance to the intersection
distance = st.number_input("Enter distance to the traffic light (meters):", min_value=0.0, value=100.0)

# Compute recommended speed
recommended_speed = suggest_speed(phase, time_remaining, distance)
st.markdown(f"### Recommended Speed: **{recommended_speed:.2f} km/h**")

# Optional: Auto-refresh control (if you want auto-refresh, see Option 2 below)
refresh_interval = st.slider("Auto-refresh interval (seconds)", min_value=1, max_value=10, value=3)
st.write("The simulation will refresh when you change the interval or manually refresh the page.")

# --- Option 1: Remove auto-refresh ---
# (Simply remove any auto-refresh code so the simulation updates when inputs change.)

# --- Option 2: Use st_autorefresh (uncomment the following block to enable auto-refresh) ---
# from streamlit_autorefresh import st_autorefresh
# st_autorefresh(interval=refresh_interval * 1000, limit=None, key="auto-refresh")
