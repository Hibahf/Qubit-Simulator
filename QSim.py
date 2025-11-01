import numpy as np
import pandas as pd
import random
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import HGate, XGate, ZGate, CXGate

def applyGate(statevector, gate, targetQubits, numQubits):
    if hasattr(gate, 'to_matrix'):
        gate = gate.to_matrix()
    if len(targetQubits) == 1:
        full_gate = 1
        for i in range(numQubits):
            full_gate = np.kron(full_gate, gate if i == targetQubits[0] else np.eye(2))
        return full_gate @ statevector
    elif len(targetQubits) == 2:
        full_gate = 1
        for i in range(numQubits):
            if i == targetQubits[0]:
                full_gate = np.kron(full_gate, np.eye(2))
            elif i == targetQubits[1]:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        return full_gate @ statevector
    else:
        raise ValueError("Gate must target 1 or 2 qubits only")

def randomGate(numQubits):
    single_qubit_gates = [HGate(), XGate(), ZGate()]
    two_qubit_gates = [CXGate()]
    if random.random() < 0.7:
        gate = random.choice(single_qubit_gates)
        target = [random.randint(0, numQubits - 1)]
    else:
        gate = random.choice(two_qubit_gates)
        target = random.sample(range(numQubits), 2)
    return gate, target

def randomCircuit(numQubits, depth):
    qc = QuantumCircuit(numQubits)
    for _ in range(depth):
        gate, targets = randomGate(numQubits)
        qc.append(gate, targets)
    return qc

class QuantumFramework:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        self.noise_model = self._makeNoiseModel()
        self.backend = Aer.get_backend("statevector_simulator")

    def _makeNoiseModel(self):
        nm = NoiseModel()
        error = depolarizing_error(self.noise_level, 1)
        nm.add_all_qubit_quantum_error(error, ['x', 'h', 'z'])
        return nm

    def simulateQuantum(self, qc):
        job = execute(qc, self.backend)
        result = job.result()
        return result.get_statevector(qc)

    def simulateNoisy(self, qc):
        noisy_qc = qc.copy()
        for qubit in range(qc.num_qubits):
            if random.random() < self.noise_level:
                noisy_qc.x(qubit)
        return self.simulateQuantum(noisy_qc)

    def simulateClassical(self, numQubits, depth):
        state = np.zeros(2 ** numQubits)
        state[0] = 1
        for _ in range(depth):
            state = np.roll(state, 1)
        return state

    def benchmark(self, numQubits=3, depth=5):
        qc = randomCircuit(numQubits, depth)
        start = time.time()
        qState = self.simulateQuantum(qc)
        qTime = time.time() - start
        start = time.time()
        cState = self.simulateClassical(numQubits, depth)
        cTime = time.time() - start
        fidelity = np.abs(np.dot(np.conjugate(cState), qState)) ** 2
        df = pd.DataFrame([{
            "Qubits": numQubits,
            "Depth": depth,
            "Quantum Time (s)": qTime,
            "Classical Time (s)": cTime,
            "Fidelity": fidelity
        }])
        return df, qc, qState

if __name__ == "__main__":
    framework = QuantumFramework(noise_level=0.02)
    df, qc, state = framework.benchmark(numQubits=3, depth=6)
    print("\nRandom Quantum Circuit:\n")
    print(qc)
    print("\nFinal Statevector (trimmed):", np.round(state[:6], 3), "...\n")
    print("Benchmark Results:\n", df.to_string(index=False))
