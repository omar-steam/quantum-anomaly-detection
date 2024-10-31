# Quantum Anomaly Detection
A quantum machine learning approach for detecting network anomalies using Qiskit and the KDD Cup dataset. This implementation uses quantum circuits for binary classification of network traffic as either normal or anomalous.

## Features
- Quantum circuit-based anomaly detection
- Data preprocessing with PCA dimensionality reduction
- Enhanced quantum circuit design with entanglement layers
- Performance optimization for faster training
- Support for the KDD Cup network security dataset

## Project Structure
```
quantum-network-anomaly/
├── README.md
├── src/
│   ├── preprocessing.py
│   ├── quantum_circuit.py
│   ├── trainer.py
│   └── utils.py
├── notebook/
│   └── quantum_anomaly_detection.ipynb
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```
git clone https://github.com/omar-steam/quantum-network-anomaly.git
cd quantum-network-anomaly
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Download the KDD Cup dataset and place it in the `data/` directory.
2. Run the preprocessing script:
```
python src/preprocessing.py
```
3. Train the model:
```
python src/trainer.py
```

## Requirements
- Python 3.8+
- Qiskit
- Qiskit Machine Learning
- NumPy
- Pandas
- Scikit-learn
- SciPy

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you want to change.

## License
[MIT](https://choosealicense.com/licenses/mit/) 

