import numpy as np
import pandas as pd
import os
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import io
from PIL import Image
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import classification_report, confusion_matrix

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

# Constants
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

# New function for label generation based on maximum altitude
def categorize_launch(altitude):
    if altitude < 50000:
        return 0  # Low altitude
    elif altitude < 100000:
        return 1  # Medium altitude
    else:
        return 2  # High altitude

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

def generate_launch_image(launch_data):
    plt.figure(figsize=(10, 6))
    for column in launch_data.columns:
        if column != 'Time':
            plt.plot(launch_data['Time'], launch_data[column], label=column)
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right')
    
    # Convert the plot to an image in memory without displaying or saving it
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

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

# Example for using the trained model for new predictions
def predict_new_launch(model, new_launch_data):
    new_launch_image = generate_launch_image(normalize_dataset(new_launch_data))
    new_launch_image = np.expand_dims(new_launch_image, axis=0)  # Reshape for model input
    prediction = model.predict(new_launch_image)
    predicted_class = np.argmax(prediction)
    # Translate predicted_class to meaningful category
    return predicted_class

dataset_path = 'datasets/rocket_launches.csv'

try:
    # Create or load the dataset
    if os.path.exists(dataset_path):
        dataset = pd.read_csv(dataset_path)
    else:
        dataset = pd.DataFrame()

    # Generate, validate, normalize, and visualize one launch data
    launch_data = simulate_rocket_launch()
    if launch_data['Altitude'].min() < 0 or launch_data['Velocity'].min() < 0:
        raise ValueError("Simulation contains negative altitudes or velocities.")
    launch_data_normalized = normalize_dataset(launch_data)
    launch_id = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    launch_data_normalized['LaunchID'] = launch_id
    launch_image_path = visualize_launch(launch_data, launch_id)
    launch_data_normalized['LaunchImage'] = launch_image_path
    dataset = pd.concat([dataset, launch_data_normalized], ignore_index=True)
    dataset.to_csv(dataset_path, index=False)
    logger.info(f"Updated dataset saved at {dataset_path}.")

    np.random.seed(42)  # Ensure reproducibility
    random.seed(42)

    num_simulations = 100  # Define the number of simulations
    launch_labels = []  # Initialize an empty list for labels

    # Generate launch data, labels, and images for CNN
    launch_images = []
    for _ in range(num_simulations):
        launch_data = simulate_rocket_launch()
        normalized_data = normalize_dataset(launch_data)
        label = categorize_launch(launch_data['Altitude'].max())
        launch_labels.append(label)
        image = generate_launch_image(normalized_data)
        launch_images.append(image)

    launch_images = np.array(launch_images)
    launch_labels = to_categorical(launch_labels)  # Convert labels to categorical

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(launch_images, launch_labels, test_size=0.2, random_state=42)

    # Define a simple CNN architecture, compile, train, and evaluate
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Continue training the CNN
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {test_acc}")

    # After model training
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Log detailed performance metrics
    logger.info("Classification Report:\n" + classification_report(y_true_classes, y_pred_classes))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_true_classes, y_pred_classes)))

    # Save the trained model
    model.save('rocket_launch_cnn_model.h5')
    logger.info("CNN model saved successfully.")

except Exception as e:
    logger.error("Fatal error in main program", exc_info=True)
    raise
