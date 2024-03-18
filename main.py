import numpy as np
import pandas as pd
import os

# Constants
g = 9.81  # Gravity acceleration (m/s^2)
rho_0 = 1.225  # Sea level air density (kg/m^3)
H = 8500  # Scale height for Earth's atmosphere (m)
Cd = 0.5  # Drag coefficient
A = 10  # Rocket cross-sectional area (m^2)
mass_empty = 50000  # Empty mass of the rocket (kg)
mass_fuel_initial = 150000  # Initial fuel mass (kg)
burn_rate = 250  # Fuel burn rate (kg/s)
thrust = 1.5e6  # Constant thrust (N)
dt = 0.1  # Time step for simulation (s)

# Function to simulate rocket launch and generate data
def simulate_rocket_launch():
    times, altitudes, velocities, masses = ([] for i in range(4))
    altitude, velocity, mass_total, time = (0, 0, mass_empty + mass_fuel_initial, 0)

    while mass_total > mass_empty and altitude >= 0:
        rho = rho_0 * np.exp(-altitude / H)
        drag = 0.5 * rho * velocity**2 * Cd * A if velocity > 0 else 0
        mass_total = max(mass_empty, mass_total - burn_rate * dt)
        force_net = thrust - drag - mass_total * g
        acceleration = force_net / mass_total
        velocity += acceleration * dt
        altitude += velocity * dt if altitude + velocity * dt > 0 else 0
        time += dt
        times.append(time)
        altitudes.append(altitude)
        velocities.append(velocity)
        masses.append(mass_total)
    
    return pd.DataFrame({'Time': times, 'Altitude': altitudes, 'Velocity': velocities, 'Mass': masses})

# Path to the dataset file
dataset_path = 'rocket_launches.csv'

# Check for existing dataset and load or initialize
if os.path.exists(dataset_path):
    dataset = pd.read_csv(dataset_path)
else:
    dataset = pd.DataFrame()

# Perform the simulation
launch_data = simulate_rocket_launch()

# Label the current launch with a unique identifier (e.g., timestamp)
launch_id = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
launch_data['LaunchID'] = launch_id

# Append the new launch data to the dataset
dataset = pd.concat([dataset, launch_data], ignore_index=True)

# Save the updated dataset
dataset.to_csv(dataset_path, index=False)

print(f"Updated dataset saved at {dataset_path}")
