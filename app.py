import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import random
import time
import plotly.express as px
from cuopt_sh_client import CuOptServiceSelfHostClient

# --- CONFIGURATION ---
st.set_page_config(page_title="Swedish EV Routing", layout="wide")

# Initialize Session State to prevent "blinking"
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'loc_df' not in st.session_state:
    st.session_state.loc_df = None
if 'dist_matrix' not in st.session_state:
    st.session_state.dist_matrix = None
if 'status_msg' not in st.session_state:
    st.session_state.status_msg = ""

# Central Hub (Depot)
DEPOT_LOCATION = {"name": "Hub (Jönköping)", "lat": 57.7826, "lon": 14.1618}

# Cities
CITIES = [
    {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"name": "Gothenburg", "lat": 57.7089, "lon": 11.9746},
    {"name": "Malmö", "lat": 55.6045, "lon": 13.0038},
    {"name": "Uppsala", "lat": 59.8586, "lon": 17.6389},
    {"name": "Västerås", "lat": 59.6173, "lon": 16.5422},
    {"name": "Örebro", "lat": 59.2753, "lon": 15.2134},
    {"name": "Linköping", "lat": 58.4108, "lon": 15.6214},
    {"name": "Norrköping", "lat": 58.5877, "lon": 16.1924},
]

# --- HELPER FUNCTIONS ---

def generate_mock_locations(num_orders=20):
    locations = []
    locations.append({
        "id": 0, "name": DEPOT_LOCATION["name"], 
        "lat": DEPOT_LOCATION["lat"], "lon": DEPOT_LOCATION["lon"], 
        "type": "Depot", "demand": 0
    })
    for i in range(num_orders):
        city = random.choice(CITIES)
        lat = city["lat"] + random.uniform(-0.1, 0.1)
        lon = city["lon"] + random.uniform(-0.1, 0.1)
        locations.append({
            "id": i + 1, "name": f"Order {i+1} ({city['name']})",
            "lat": lat, "lon": lon, 
            "type": "Delivery", "demand": random.randint(2, 5)
        })
    return pd.DataFrame(locations)

def calculate_distance_matrix(df):
    n = len(df)
    matrix = np.zeros((n, n))
    coords = list(zip(df['lat'], df['lon']))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = geodesic(coords[i], coords[j]).kilometers
    return matrix.tolist()

def solve_routing(locations_df, matrix, num_vehicles, max_distance, vehicle_capacity):
    vehicle_ids = [f"Truck-{i+1}" for i in range(num_vehicles)]
    
    data = {
        "cost_matrix_data": {"data": {"1": matrix}},
        "fleet_data": {
            "vehicle_locations": [[0, 0]] * num_vehicles,
            "vehicle_ids": vehicle_ids,
            "vehicle_types": [1] * num_vehicles,
            "capacities": [[vehicle_capacity] * num_vehicles],
            "vehicle_max_costs": [max_distance] * num_vehicles 
        },
        "task_data": {
            "task_locations": list(range(1, len(locations_df))), 
            "demand": [locations_df['demand'].iloc[1:].tolist()]
        },
        "solver_config": {"time_limit": 0.5}
    }

    try:
        client = CuOptServiceSelfHostClient(ip="localhost", port=5000)
        solution = client.get_optimized_routes(data)
        
        if solution and "response" in solution:
            if solution["response"]["solver_response"]["status"] == 0:
                return solution, "✅ Optimized Successfully"
            else:
                return solution, "⚠️ Solution Found (Constraints Tight)"
        return None, "❌ Solver returned empty response"
    except Exception as e:
        return None, f"❌ Connection Error: {str(e)}"

# --- UI LAYOUT ---

st.title("🚛 Swedish EV Logistics Planner")

with st.sidebar:
    st.header("1. Fleet Settings")
    num_vehicles = st.slider("Fleet Size", 3, 15, 8)
    # Default raised to 1200km to ensure feasibility in Sweden
    max_range = st.slider("Max Range (km)", 200, 2000, 1200) 
    capacity = st.slider("Capacity (Pallets)", 10, 100, 50)
    
    st.header("2. Demand")
    num_orders = st.slider("Number of Orders", 10, 100, 40)
    
    if st.button("Generate & Solve", type="primary"):
        with st.spinner("Solving..."):
            # Generate Data
            loc_df = generate_mock_locations(num_orders)
            dist_matrix = calculate_distance_matrix(loc_df)
            
            # Store inputs in session state
            st.session_state.loc_df = loc_df
            st.session_state.dist_matrix = dist_matrix
            
            # Solve
            sol, msg = solve_routing(loc_df, dist_matrix, num_vehicles, max_range, capacity)
            st.session_state.solution = sol
            st.session_state.status_msg = msg

# --- RENDER RESULTS ---

if st.session_state.loc_df is not None:
    # Status Message
    if "✅" in st.session_state.status_msg:
        st.success(st.session_state.status_msg)
    else:
        st.warning(st.session_state.status_msg)

    # Prepare Map
    m = folium.Map(location=[58.0, 14.5], zoom_start=6, tiles="cartodbpositron")
    
    # Draw Depot
    folium.Marker(
        [DEPOT_LOCATION['lat'], DEPOT_LOCATION['lon']], 
        popup="DEPOT", icon=folium.Icon(color='black', icon='home')
    ).add_to(m)

    # Parse Solution if available
    if st.session_state.solution:
        routes_data = st.session_state.solution["response"]["solver_response"].get("vehicle_data", {})
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']
        
        # Draw Routes
        for idx, (v_id, v_data) in enumerate(routes_data.items()):
            route = v_data.get("route", [])
            if len(route) > 2:
                points = [
                    (st.session_state.loc_df.iloc[i]['lat'], st.session_state.loc_df.iloc[i]['lon']) 
                    for i in route
                ]
                color = colors[idx % len(colors)]
                
                # Line
                folium.PolyLine(points, color=color, weight=4, opacity=0.7, tooltip=f"Truck {v_id}").add_to(m)
                
                # Stops
                for p_idx in route[1:-1]:
                    row = st.session_state.loc_df.iloc[p_idx]
                    folium.CircleMarker(
                        [row['lat'], row['lon']], radius=5, color=color, fill=True,
                        popup=f"{row['name']} ({row['demand']} pallets)"
                    ).add_to(m)
    else:
        # Fallback: Draw raw points if no solution
        for _, row in st.session_state.loc_df.iloc[1:].iterrows():
            folium.CircleMarker(
                [row['lat'], row['lon']], radius=3, color='gray',
                popup=row['name']
            ).add_to(m)

    st_folium(m, height=600, width=1000)

    # Metrics
    if st.session_state.solution:
        resp = st.session_state.solution["response"]["solver_response"]
        dropped = resp.get("dropped_tasks", {}).get("task_id", [])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Fleet Distance", f"{int(resp.get('solution_cost', 0))} km")
        c2.metric("Vehicles Used", resp.get("num_vehicles", 0))
        c3.metric("Dropped Orders", len(dropped), delta_color="inverse")
else:
    st.info("👈 Click **Generate & Solve** in the sidebar to start.")