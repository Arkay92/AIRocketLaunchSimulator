import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import logging
import argparse
import matplotlib.pyplot as plt

# Setup advanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('rocket_simulation.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Ensure the output directories exist
for path in ['launch_plots', 'datasets']:
    os.makedirs(path, exist_ok=True)

# Command-line arguments for parameter customization
parser = argparse.ArgumentParser(description="Rocket Launch Simulation Parameters")
parser.add_argument("--burn_rate", type=float, default=250, help="Fuel burn rate (kg/s)")
parser.add_argument("--thrust", type=float, default=1.5e6, help="Constant thrust (N)")
args = parser.parse_args()

# Constants (some are now adjustable via command-line arguments)
g = 9.81  # Gravity acceleration (m/s^2)
rho_0 = 1.225  # Sea level air density (kg/m^3)
H = 8500  # Scale height for Earth's atmosphere (m)
Cd = 0.5  # Drag coefficient
A = 10  # Rocket cross-sectional area (m^2)
mass_empty = 50000  # Empty mass of the rocket (kg)
mass_fuel_initial = 150000  # Initial fuel mass (kg)
burn_rate = args.burn_rate  # Fuel burn rate (kg/s), now customizable
thrust = args.thrust  # Constant thrust (N), now customizable
dt = 0.1  # Time step for simulation (s)

def simulate_rocket_launch():
    times, altitudes, velocities, masses, accelerations = ([] for i in range(5))
    altitude, velocity, mass_total, time, previous_velocity = (0, 0, mass_empty + mass_fuel_initial, 0, 0)

    try:
        while mass_total > mass_empty and altitude >= 0:
            rho = rho_0 * np.exp(-altitude / H)
            drag = 0.5 * rho * velocity**2 * Cd * A if velocity > 0 else 0
            mass_total = max(mass_empty, mass_total - burn_rate * dt)
            force_net = thrust - drag - mass_total * g
            acceleration = force_net / mass_total
            velocity += acceleration * dt
            altitude += velocity * dt if altitude + velocity * dt > 0 else 0
            time += dt
            current_acceleration = (velocity - previous_velocity) / dt
            times.append(time)
            altitudes.append(altitude)
            velocities.append(velocity)
            masses.append(mass_total)
            accelerations.append(current_acceleration)
            previous_velocity = velocity
    except Exception as e:
        logger.error("Simulation error", exc_info=True)
        raise

    return pd.DataFrame({'Time': times, 'Altitude': altitudes, 'Velocity': velocities, 'Mass': masses, 'Acceleration': accelerations})

def normalize_dataset(df):
    try:
        scaler = StandardScaler()
        df[['Time', 'Altitude', 'Velocity', 'Mass', 'Acceleration']] = scaler.fit_transform(df[['Time', 'Altitude', 'Velocity', 'Mass', 'Acceleration']])
    except Exception as e:
        logger.error("Normalization error", exc_info=True)
        raise
    return df

def visualize_launch(launch_data, launch_id):
    plt.figure(figsize=(10, 6))
    plt.plot(launch_data['Time'], launch_data['Altitude'], label='Altitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title(f'Rocket Altitude Over Time (Launch ID: {launch_id})')
    plt.legend()
    image_path = f"launch_plots/{launch_id}.png"
    plt.savefig(image_path)
    plt.close()  # Close the plot to free up memory
    return image_path

dataset_path = 'datasets/rocket_launches.csv'

try:
    if os.path.exists(dataset_path):
        dataset = pd.read_csv(dataset_path)
    else:
        dataset = pd.DataFrame()

    launch_data = simulate_rocket_launch()

    # Data validation
    if launch_data['Altitude'].min() < 0 or launch_data['Velocity'].min() < 0:
        raise ValueError("Simulation contains negative altitudes or velocities, indicating a problem with the launch parameters.")

    launch_data_normalized = normalize_dataset(launch_data)

    launch_id = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    launch_data_normalized['LaunchID'] = launch_id

    # Visualize and save the launch plot
    launch_image_path = visualize_launch(launch_data, launch_id)
    launch_data_normalized['LaunchImage'] = launch_image_path

    dataset = pd.concat([dataset, launch_data_normalized], ignore_index=True)

    dataset.to_csv(dataset_path, index=False)
    logger.info(f"Updated dataset saved at {dataset_path}.")

except Exception as e:
    logger.error("Fatal error in main program", exc_info=True)
    raise
