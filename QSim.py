import numpy as np
import random
import pandas as pd
from time import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, NoiseModel
from qiskit.algorithms import Shor
from qiskit.primitives import Sampler
from qiskit_aer.noise import depolarizing_error
from qiskit.circuit.library import HGate, XGate, ZGate

sampler = Sampler()
shor = Shor(sampler=sampler)

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

sqgate = [HGate(), XGate(), ZGate()]

simulator = AerSimulator(method='statevector')

def initializeState(qubitamt):
    state = np.zeros(2**qubitamt, dtype=complex)
    state[0] = 1.0
    return state

def applyGate(state, gate, target, qubitamt):
    if hasattr(gate, 'to_matrix'):
        gate_matrix = gate.to_matrix()
    else:
        gate_matrix = gate
        
    eyeBefore = np.eye(2**target)
    eyeAfter = np.eye(2**(qubitamt - target - 1))
    full = np.kron(np.kron(eyeBefore, gate_matrix), eyeAfter)
    return full @ state

def applyCNOT(state, control, target, qubitamt):
    qc = QuantumCircuit(qubitamt)
    
    qc.cx(control, target)
    qc.save_statevector()
    
    result = simulator.run(qc).result()
    return result.get_statevector(qc)

def randomGate(state, qubitamt):
    gate_type = random.choice(['single', 'two'])
    if gate_type == 'single' and qubitamt >= 1:
        gate = random.choice(sqgate)
        target = random.randint(0, qubitamt - 1)
        state = applyGate(state, gate, target, qubitamt)
    elif gate_type == 'two' and qubitamt >= 2:
        control, target = random.sample(range(qubitamt), 2)
        state = applyCNOT(state, control, target, qubitamt)
    return state

def randomCircuit(qubitamt, depth):
    qc = QuantumCircuit(qubitamt)
    for _ in range(depth):
        if random.choice([True, False]) and qubitamt >= 2:
            control, target = random.sample(range(qubitamt), 2)
            qc.cx(control, target)
        else:
            gate = random.choice([HGate(), XGate(), ZGate()])
            target = random.randint(0, qubitamt - 1)
            qc.append(gate, [target])
    
    qc.measure_all()
    result = simulator.run(qc).result()
    state = result.get_statevector(qc)
    return qc, state

def noise(qc, errorProb=0.05):
    noisy_qc = qc.copy()
    for qubit in range(len(noisy_qc.qubits)):
        if random.random() < errorProb:
            noisy_qc.x(qubit)
    return noisy_qc

def results():
    data = {
        'Qubits': [2, 3, 4],
        'Classical Time (ms)': [120, 980, 10500],
        'Quantum Time (ms)': [45, 210, 1100]
    }
    print("Example data only - replace with benchmark() output:")
    print(pd.DataFrame(data).to_latex(index=False))

class QuantumFramework:
    def initialize(self):
        self.simulator = AerSimulator(method='statevector')
        self.noise_model = self.createNoise()
        self.sampler = Sampler()
    
    def createNoise(self):
        noise_model = NoiseModel()
        error1q = depolarizing_error(0.05, 1)
        error2q = depolarizing_error(0.1, 2)
        noise_model.add_all_qubit_quantum_error(error1q, ['h', 'x', 'z'])
        noise_model.add_all_qubit_quantum_error(error2q, ['cx'])
        return noise_model
    
    def runShors(self, N):
        shor = Shor(sampler=self.sampler)
        result = shor.factor(N)
        return result
    
    def benchmark(self, maxQubits=4):
        data = []
        for n in range(2, maxQubits + 1):
            for depth in [5, 10, 20]:
                # Quantum approach (using Qiskit)
                qc, state_ideal = randomCircuit(n, depth=depth)
                t0 = time()
                transpiled = transpile(qc, self.simulator)
                job = self.simulator.run(transpiled)
                result = job.result()
                qTime = time() - t0
                
                # Classical approach (using your manual methods)
                t0 = time()
                state = initializeState(n)
                for _ in range(depth):
                    state = randomGate(state, n)
                cTime = time() - t0

                data.append({
                    'Qubits': n, 
                    "Depth": depth,
                    'Quantum Time in milliseconds': qTime * 1000,
                    'Classical Time in milliseconds': cTime * 1000,
                    'Speedup': cTime / qTime if qTime > 0 else float('inf')
                })
        return pd.DataFrame(data)

if __name__ == "__main__":
    qc, final = randomCircuit(qubitamt=2, depth=5)
    probabilities = np.abs(final)**2
    print("Final state probabilities:", probabilities)
    print("Circuit:")
    print(qc)
    
    framework = QuantumFramework()
    framework.initialize()
    
    df = framework.benchmark(maxQubits=3)
    print("\nBenchmark Results:")
    print(df)
    
    print("\nLaTeX Table:")
    print(df.to_latex(index=False, float_format="%.2f"))
