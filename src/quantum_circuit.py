from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_enhanced_circuit(n_qubits):
    # Create an enhanced quantum circuit with entanglement layers.
    circuit = QuantumCircuit(n_qubits)
    
    # First rotation layer
    for i in range(n_qubits):
        circuit.h(i)
        circuit.ry(Parameter(f'theta_{i}'), i)

    # Entanglement layer
    for i in range(n_qubits-1):
        circuit.cx(i, i+1)

    # Second rotation layer
    for i in range(n_qubits):
        circuit.ry(Parameter(f'phi_{i}'), i)

    return circuit 
