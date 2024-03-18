# RocketLaunchSimulator

Simulate and analyze rocket launches with Python. This project models the ascent of a rocket, taking into account thrust, drag, gravity, and fuel consumption, and outputs the rocket's altitude, velocity, and mass over time.

## Features

- Simulate rocket launches using basic physics principles.
- Generate datasets for each launch, with the ability to start fresh or append to existing data.
- Visualize the trajectory, speed, and mass change of the rocket over time.

## Installation

Clone this repository to your local machine:
```
git clone https://github.com/Arkay92/RocketLaunchSimulator.git
```

Navigate to the project directory:
```
cd RocketLaunchSimulator
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the simulation script:
```
python main.py
```

The script will generate a dataset (`rocket_launches.csv`) in the project directory, containing the simulation data. Each launch is uniquely identified and appended to the dataset if it already exists.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is open source and available under the [MIT License](LICENSE).
