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
st.set_page_config(page_title="Swedish EV Routing (Education Mode)", layout="wide")

# Initialize Session State
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'loc_df' not in st.session_state:
    st.session_state.loc_df = None
if 'station_df' not in st.session_state:
    st.session_state.station_df = None
if 'dist_matrix' not in st.session_state:
    st.session_state.dist_matrix = None
if 'status_msg' not in st.session_state:
    st.session_state.status_msg = ""

# Central Hub (Depot)
DEPOT_LOCATION = {"name": "Hub (Jönköping)", "lat": 57.7826, "lon": 14.1618}

# Major Cities for Simulation
CITIES = [
    {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"name": "Gothenburg", "lat": 57.7089, "lon": 11.9746},
    {"name": "Malmö", "lat": 55.6045, "lon": 13.0038},
    {"name": "Uppsala", "lat": 59.8586, "lon": 17.6389},
    {"name": "Västerås", "lat": 59.6173, "lon": 16.5422},
    {"name": "Örebro", "lat": 59.2753, "lon": 15.2134},
    {"name": "Linköping", "lat": 58.4108, "lon": 15.6214},
    {"name": "Helsingborg", "lat": 56.0465, "lon": 12.6945},
    {"name": "Norrköping", "lat": 58.5877, "lon": 16.1924},
    {"name": "Borås", "lat": 57.7210, "lon": 12.9401},
]

# --- DATA GENERATION FUNCTIONS ---

def generate_mock_locations(num_orders=20):
    locations = []
    # Depot is always ID 0
    locations.append({
        "id": 0, "name": DEPOT_LOCATION["name"], 
        "lat": DEPOT_LOCATION["lat"], "lon": DEPOT_LOCATION["lon"], 
        "type": "Depot", "demand": 0
    })
    for i in range(num_orders):
        city = random.choice(CITIES)
        lat = city["lat"] + random.uniform(-0.15, 0.15)
        lon = city["lon"] + random.uniform(-0.15, 0.15)
        locations.append({
            "id": i + 1, "name": f"Order {i+1} ({city['name']})",
            "lat": lat, "lon": lon, 
            "type": "Delivery", "demand": random.randint(2, 5)
        })
    return pd.DataFrame(locations)

def generate_charging_stations(num_stations=15):
    """Creates mock charging stations scattered across the region."""
    stations = []
    for i in range(num_stations):
        # Create stations somewhat randomly but within the bounding box of cities
        lat = random.uniform(56.0, 59.5) # South to Central Sweden
        lon = random.uniform(12.0, 18.0) # West to East
        stations.append({
            "id": f"S-{i+1}",
            "name": f"SuperCharger {i+1}",
            "lat": lat, "lon": lon,
            "type": "Station"
        })
    return pd.DataFrame(stations)

def calculate_distance_matrix(df):
    n = len(df)
    matrix = np.zeros((n, n))
    coords = list(zip(df['lat'], df['lon']))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = geodesic(coords[i], coords[j]).kilometers
    return matrix.tolist()

# --- OPTIMIZATION LOGIC ---

def solve_routing(locations_df, matrix, num_vehicles, vehicle_capacity):
    """
    Step 1: Use NVIDIA cuOpt to solve the SEQUENCE of deliveries.
    We purposely give infinite range here so cuOpt focuses on the best path.
    """
    vehicle_ids = [f"Truck-{i+1}" for i in range(num_vehicles)]
    
    data = {
        "cost_matrix_data": {"data": {"1": matrix}},
        "fleet_data": {
            "vehicle_locations": [[0, 0]] * num_vehicles,
            "vehicle_ids": vehicle_ids,
            "vehicle_types": [1] * num_vehicles,
            "capacities": [[vehicle_capacity] * num_vehicles],
            # We relax max cost here to allow route-first, charge-second logic
            "vehicle_max_costs": [5000] * num_vehicles 
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
        
        if solution and "response" in solution and solution["response"]["solver_response"]["status"] == 0:
            return solution, "✅ Optimized Successfully"
        
        # Fallback simulation if solver fails/unreachable
        return mock_fallback_solver(num_vehicles, len(locations_df)-1), "⚠️ Simulation Mode (Solver Unreachable)"
        
    except Exception as e:
        return mock_fallback_solver(num_vehicles, len(locations_df)-1), f"⚠️ Simulation Mode (Error: {str(e)})"

def mock_fallback_solver(num_vehicles, num_orders):
    # Creates fake routes for demo purposes
    vehicle_data = {}
    available_orders = list(range(1, num_orders + 1))
    random.shuffle(available_orders)
    chunks = np.array_split(available_orders, num_vehicles)
    for i in range(num_vehicles):
        route = [0] + list(chunks[i]) + [0]
        vehicle_data[str(i)] = {"route": [int(x) for x in route], "cost": 0}
    return {"response": {"solver_response": {"status": 0, "vehicle_data": vehicle_data}}}

def inject_charging_stops(original_routes, locations_df, stations_df, max_range):
    """
    Step 2: Post-Processing Heuristic.
    Walks the route. If accumulated distance > range, finds nearest charger, 
    inserts it into route, and resets battery.
    """
    optimized_routes = {}
    
    for v_id, v_data in original_routes.items():
        raw_indices = v_data["route"]
        if len(raw_indices) <= 2: continue # Skip empty
        
        new_route = []
        current_battery_used = 0
        total_dist_with_charging = 0
        
        # Start at depot
        curr_loc_idx = raw_indices[0]
        new_route.append({"type": "Depot", "data": locations_df.iloc[curr_loc_idx]})
        
        for next_loc_idx in raw_indices[1:]:
            # Calculate distance to next stop
            p1 = (locations_df.iloc[curr_loc_idx]['lat'], locations_df.iloc[curr_loc_idx]['lon'])
            p2 = (locations_df.iloc[next_loc_idx]['lat'], locations_df.iloc[next_loc_idx]['lon'])
            leg_dist = geodesic(p1, p2).kilometers
            
            # CHECK: Do we have range?
            if current_battery_used + leg_dist > max_range:
                # LOW BATTERY! Find nearest charger from current location
                best_station = None
                min_detour = float('inf')
                
                for _, station in stations_df.iterrows():
                    s_coords = (station['lat'], station['lon'])
                    # Dist from current -> Station
                    d1 = geodesic(p1, s_coords).kilometers
                    # Dist from Station -> Next Stop
                    d2 = geodesic(s_coords, p2).kilometers
                    
                    # Heuristic: Find station that minimizes total detour
                    if d1 < max_range - current_battery_used: # Must be reachable
                        detour = d1 + d2
                        if detour < min_detour:
                            min_detour = detour
                            best_station = station
                            dist_to_station = d1
                            dist_from_station = d2

                if best_station is not None:
                    # Insert Station
                    new_route.append({"type": "Station", "data": best_station})
                    current_battery_used = 0 # RECHARGE!
                    total_dist_with_charging += dist_to_station
                    
                    # Now we are at station, update current state for the next leg (Station -> Dest)
                    # We virtually moved to station, so next leg is Station -> Next Node
                    current_battery_used += dist_from_station 
                    total_dist_with_charging += dist_from_station
                else:
                    # Stranded! (In a real app, this is an alert)
                    pass 
            else:
                # Drive normally
                current_battery_used += leg_dist
                total_dist_with_charging += leg_dist
            
            # Arrive at destination
            new_route.append({"type": "Stop", "data": locations_df.iloc[next_loc_idx]})
            curr_loc_idx = next_loc_idx
            
        optimized_routes[v_id] = {
            "path": new_route,
            "total_dist": total_dist_with_charging
        }
        
    return optimized_routes

# --- UI LAYOUT ---

st.title("⚡ Swedish EV Logistics: Route & Charge")

with st.sidebar:
    st.header("1. Fleet & Battery")
    num_vehicles = st.slider("Fleet Size", 2, 10, 5)
    max_range = st.slider("Max Range (km)", 100, 600, 300, help="Distance before charging is required")
    capacity = st.slider("Cargo Capacity", 10, 50, 30)
    
    st.header("2. Infrastructure")
    num_stations = st.slider("Num. Charging Stations", 5, 50, 20)
    num_orders = st.slider("Num. Orders", 10, 50, 25)
    
    if st.button("Generate & Optimize", type="primary"):
        with st.spinner("Optimizing Topology..."):
            # 1. Generate World
            loc_df = generate_mock_locations(num_orders)
            station_df = generate_charging_stations(num_stations)
            dist_matrix = calculate_distance_matrix(loc_df)
            
            # 2. Save State
            st.session_state.loc_df = loc_df
            st.session_state.station_df = station_df
            
            # 3. Solve Base Routes (VRP)
            raw_sol, msg = solve_routing(loc_df, dist_matrix, num_vehicles, capacity)
            
            # 4. Inject Charging Logic
            if raw_sol and "response" in raw_sol:
                raw_routes = raw_sol["response"]["solver_response"]["vehicle_data"]
                final_routes = inject_charging_stops(raw_routes, loc_df, station_df, max_range)
                st.session_state.solution = final_routes
                st.session_state.status_msg = msg

# --- VISUALIZATION ---

if st.session_state.solution is not None:
    st.success(st.session_state.status_msg)
    
    # Metrics
    total_km = sum(r['total_dist'] for r in st.session_state.solution.values())
    total_charges = sum(1 for r in st.session_state.solution.values() for stop in r['path'] if stop['type'] == 'Station')
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Fleet Distance", f"{int(total_km)} km")
    m2.metric("Charging Stops", total_charges)
    m3.metric("Avg. Distance / Truck", f"{int(total_km / num_vehicles)} km")

    # Map
    m = folium.Map(location=[58.0, 14.5], zoom_start=6, tiles="cartodbpositron")
    
    # 1. Plot Infrastructure
    # Depot
    folium.Marker(
        [DEPOT_LOCATION['lat'], DEPOT_LOCATION['lon']], 
        popup="DEPOT", icon=folium.Icon(color='black', icon='home')
    ).add_to(m)
    
    # Charging Stations (Show all available)
    for _, s in st.session_state.station_df.iterrows():
        folium.CircleMarker(
            [s['lat'], s['lon']], radius=3, color='green', fill=True, 
            popup=s['name'], tooltip="Charger"
        ).add_to(m)

    # 2. Plot Routes
    colors = ['red', 'blue', 'purple', 'orange', 'darkred', 'cadetblue']
    
    for i, (v_id, data) in enumerate(st.session_state.solution.items()):
        path = data['path']
        coords = [(p['data']['lat'], p['data']['lon']) for p in path]
        color = colors[i % len(colors)]
        
        # Route Line
        folium.PolyLine(coords, color=color, weight=4, opacity=0.7, tooltip=v_id).add_to(m)
        
        # Stops
        for stop in path:
            lat, lon = stop['data']['lat'], stop['data']['lon']
            if stop['type'] == 'Delivery':
                folium.CircleMarker(
                    [lat, lon], radius=5, color=color, fill=True, fill_color='white',
                    popup=f"{stop['data']['name']}"
                ).add_to(m)
            elif stop['type'] == 'Station':
                # Highlight Used Chargers
                folium.Marker(
                    [lat, lon], 
                    icon=folium.Icon(color='green', icon='bolt', prefix='fa'),
                    popup=f"Charging: {v_id}"
                ).add_to(m)

    st_folium(m, height=600)
    
    st.markdown("### 📝 Route Manifest")
    for v_id, data in st.session_state.solution.items():
        with st.expander(f"{v_id} - {int(data['total_dist'])} km"):
            steps = []
            for step in data['path']:
                icon = "🏭" if step['type'] == 'Depot' else "⚡" if step['type'] == 'Station' else "📦"
                steps.append(f"{icon} {step['data']['name']}")
            st.write(" ➝ ".join(steps))

else:
    st.info("👈 Set fleet parameters and click **Generate & Optimize**.")