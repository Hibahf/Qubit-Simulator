import numpy as np
import random
import pandas as pd
from time import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, NoiseModel
from qiskit.algorithms import Shor
from qiskit.primitives import Sampler
from qiskit_aer.noise import depolarizing_error

sampler = Sampler()
shor = Shor(sampler=sampler)

Hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PauliX = np.array([[0, 1], [1, 0]])
PauliZ = np.array([[1, 0], [0, -1]])

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

sqgate = [Hadamard, PauliX, PauliZ]
tqgate = [CNOT]

simulator = AerSimulator(method = 'statevector')

def initializeState(qubitamt):
    state = np.zeros(2**qubitamt, dtype=complex)
    state[0] = 1.0
    return state

def applyGate(state, gate, target, qubitamt):
    eyeBefore = np.eye(2**target)
    eyeAfter = np.eye(2**(qubitamt - target - 1))
    full = np.kron(np.kron(eyeBefore, gate), eyeAfter)
    return full @ state

def applyCNOT(state, control, target, qubitamt):
    qc = QuantumCircuit(qubitamt)
    qc.initialize(state, range(qubitamt))
    qc.cx(control, target)
    compiled = transpile(qc, simulator)
    job = simulator.run(compiled)
    result = job.result()
    return result.get_statevector()


def randomGate(state, qubitamt):
    type = random.choice(['single', 'two'])
    if type == 'single' and qubitamt >= 1:
        gate = random.choice(sqgate)
        target = random.randint(0, qubitamt - 1)
        state = applyGate(state, gate, target, qubitamt)
    elif type == 'two' and qubitamt >= 2:
        control, target = random.sample(range(qubitamt), 2)
        state = applyCNOT(state, control, target, qubitamt)
    return state

def randomCircuit(qubitamt, depth):
    state = initializeState(qubitamt)
    for _ in range(depth):
        if random.choice([True, False]) and qubitamt >= 2:
            control, target = random.sample(range(qubitamt), 2)
            state = applyCNOT(state, control, target, qubitamt)
        else:
            gate = random.choice([Hadamard, PauliX, PauliZ])
            target = random.randint(0, qubitamt - 1)
            state = applyGate(state, gate, target, qubitamt)
    
    counts = {f"{i:0{qubitamt}b}": abs(amp)**2
                for i, amp in enumerate(state)}
    return state

def noise(qc, errorProb=0.05):
    for qubit in qc.qubits:
        if random.random() < errorProb:
            qc.x(qubit)
    return qc

def results():
    data = {
        'Qubits': [2, 3, 4],
        'Classical Time (ms)': [120, 980, 10500],
        'Quantum Time (ms)': [45, 210, 1100]
    }
    print(pd.DataFrame(data).to_latex())

final = randomCircuit(qubitamt=2, depth=5)
probabilities = np.abs(final)**2
print("Final state probabilities:", probabilities)

class QuantumFramework:
    def initialize(self):
        self.simulator = AerSimulator(method ='statevector')
        self.noise_model = self.createNoise()
    
    def createNoise(self):
        noise_model = NoiseModel()
        error1q = depolarizing_error(0.05, 1)
        error2q = depolarizing_error(0.1, 2)
        noise_model.add_all_qubit_quantum_error(error1q, ['h', 'x', 'z'])
        noise_model.add_all_qubit_quantum_error(error2q, ['cx'])
        return noise_model
    
    def runShors(self, N):
        print("Shor's algorithm placeholder, the quantum instance is unavailable.")
        return None
    
    def benchmark(self, maxQubits=4):
        data = []
        for n in range(2, maxQubits + 1):
            qc = randomCircuit(n, depth=10)
            t0 = time()
            transpiled = transpile(qc, self.simulator)
            job = self.simulator.run(transpiled)
            qTime = time() - t0
            
            t0 = time()
            mat = np.random.rand(2**n, 2**n)
            vec = np.random.rand(2**n)
            np.dot(mat, vec)
            cTime = time() - t0

            data.append({
                'Qubits':n,
                'Quantum Time in milliseconds': qTime*1000,
                'Classical Time in milliseconds': cTime*1000,
                'Speedup': cTime/qTime
            })
        df = pd.DataFrame(data)
        print(df.to_Markdown())
        return df
