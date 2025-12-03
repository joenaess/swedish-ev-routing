# ⚡ Swedish EV Logistics Planner

A GPU-accelerated Electric Vehicle Routing Problem (EVRP) solver powered by **NVIDIA cuOpt** and **Streamlit**.

This project demonstrates how to optimize logistics for an electric truck fleet in Sweden, balancing spatial routing constraints with battery range limits and cargo capacity. It uses a "Route-First, Charge-Second" heuristic to dynamically insert charging stops into optimized routes.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-cuOpt-green)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

## 🚀 Features

* **GPU-Accelerated Routing:** Uses NVIDIA cuOpt's high-performance routing solver to sequence deliveries across major Swedish cities (Stockholm, Gothenburg, Malmö, etc.).
* **Smart Charging Logic:** Automatically detects when a truck's predicted range is insufficient and reroutes it to the optimal charging station to "refuel."
* **Interactive Visualization:**
    * **Folium Maps:** Visualizes routes, depots, delivery stops, and active charging stations.
    * **Plotly Charts:** Analyzes fleet battery usage and cargo capacity utilization.
* **Dynamic Constraints:** Adjust fleet size, battery range (km), and cargo capacity (pallets) in real-time.

## 🛠️ Prerequisites

* **Hardware:** NVIDIA GPU (Ampere/Ada Lovelace architecture recommended, e.g., RTX 30/40 series).
* **OS:** Linux (tested on Pop!_OS) or WSL2.
* **Drivers:** NVIDIA Drivers (535+) and CUDA Toolkit 12/13.
* **Python:** Version 3.10 or higher.

## 📦 Installation

1.  **Clone or Create Directory**
    ```bash
    mkdir ~/swedish-ev-routing
    cd ~/swedish-ev-routing
    ```

2.  **Set up Virtual Environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    *Note: We point to the NVIDIA PyPI index to get the optimized `cuopt` binaries.*
    
    Create a `requirements.txt` file with:
    ```text
    streamlit
    pandas
    geopy
    folium
    streamlit-folium
    plotly
    --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com)
    cuopt-server-cu13
    cuopt-sh-client
    ```

    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

This application requires two terminal windows: one for the optimization server and one for the frontend UI.

### 1. Start the Solver Server (Terminal 1)
You must export the library path so the server can find the NVIDIA runtime libraries.

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(find .venv/lib/ -name "libnvrtc.so.*" -printf "%h\n" | head -n 1)
python3 -m cuopt_server.cuopt_service
```

Wait until you see Uvicorn running on <http://0.0.0.0:5000>.

### 2. Run the Application (Terminal 2)

```bash
source .venv/bin/activate
streamlit run app.py
```

### 3. Interact

Open your browser to <http://localhost:8501>.

* Sidebar: Adjust "Max Range" to see how the solver forces trucks to visit charging stations.

* Map: Look for the Green Bolt (⚡) icons. These represent stops where the algorithm determined a charge was necessary.

## 🧠 algorithmic Logic

This PoC solves the EV Routing Problem in two stages:

**1. Topological Optimization (cuOpt)**: We treat the problem as a standard Capacitated Vehicle Routing Problem (CVRP). We feed the distance matrix and demand to NVIDIA cuOpt, which uses parallel heuristics to find the most efficient sequence of stops to minimize total fleet distance. We purposely relax the battery constraint in this step to prevent the solver from declaring the problem "Infeasible."

**2. Energy Post-Processing (Python)**: We walk through the optimized route node-by-node, tracking the accumulated distance.

```IF``` distance to next node > remaining range:

```THEN``` search the station_df for the charger that minimizes the detour.

```INSERT``` the station into the route and reset accumulated distance (simulate 100% charge).

## 📂 Project Structure

```Plaintext
swedish-ev-routing/
├── app.py              # Main application logic (UI, Solver, Heuristics)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .venv/              # Virtual environment
```

## ⚠️ Troubleshooting

**"Feasible solutions could not be found"**: Your constraints are too tight. Try increasing Max Range or Fleet Size in the sidebar. If the orders are scattered 600km apart and your range is 300km, a solution may be mathematically impossible without more intermediate hops.

**"CuPy failed to load libnvrtc.so"**: You forgot to run the export LD_LIBRARY_PATH... command in Terminal 1 before starting the server.

**Map blinks or resets**: This is fixed in the latest version using st.session_state to persist results between reruns.

## 📄 License

This project is open for educational and testing purposes.