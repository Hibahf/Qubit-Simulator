import numpy as np
import pandas as pd
import random
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import HGate, XGate, ZGate, CXGate
 
 
def kron_n(matrices):
    res = np.array([[1.0]])
    for m in matrices:
        res = np.kron(res, m)
    return res
 
 
def apply_gate_statevector(state, gate_matrix, targets, num_qubits):
    k = len(targets)
    if k == 0:
        return state
    if k == num_qubits:
        return gate_matrix @ state
 
    state_tensor = state.reshape([2] * num_qubits)
    remaining = [i for i in range(num_qubits) if i not in targets]
    perm = list(targets) + remaining
    inv_perm = np.argsort(perm)
    permuted = np.transpose(state_tensor, axes=perm)
    dim_left = 2 ** k
    dim_right = 2 ** (num_qubits - k)
    reshaped = permuted.reshape((dim_left, dim_right))
    after = gate_matrix @ reshaped
    after_tensor = after.reshape([2] * num_qubits)
    final = np.transpose(after_tensor, axes=inv_perm)
    return final.reshape(2 ** num_qubits)
 
 
def random_gate(num_qubits):
    single_qubit_gates = [HGate(), XGate(), ZGate()]
    two_qubit_gates = [CXGate()]
    if num_qubits == 1 or random.random() < 0.7:
        gate = random.choice(single_qubit_gates)
        target = [random.randint(0, num_qubits - 1)]
        return gate, target
    else:
        gate = random.choice(two_qubit_gates)
        control = random.randint(0, num_qubits - 1)
        target = random.choice([q for q in range(num_qubits) if q != control])
        return gate, [control, target]
 
 
def random_circuit(num_qubits, depth):
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        gate, targets = random_gate(num_qubits)
        qc.append(gate, targets)
    return qc
 
 
class QuantumFramework:
    def __init__(self, noise_level=0.0):
        self.noise_level = float(noise_level)
        self._backend_sv = AerSimulator(method="statevector")
        self._backend_qasm = AerSimulator(method="density_matrix")
        self.noise_model = self._make_noise_model() if self.noise_level > 0 else None
 
    def _make_noise_model(self):
        nm = NoiseModel()
        error_1q = depolarizing_error(self.noise_level, 1)
        error_2q = depolarizing_error(min(1.0, self.noise_level * 2.0), 2)
        nm.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'z'])
        nm.add_all_qubit_quantum_error(error_2q, ['cx'])
        return nm
 
    def simulate_qiskit_statevector(self, qc: QuantumCircuit):
        from qiskit import transpile
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        tqc = transpile(qc_sv, self._backend_sv)
        result = self._backend_sv.run(tqc).result()
        return np.array(result.get_statevector(tqc))
 
    def simulate_qiskit_noisy_counts(self, qc: QuantumCircuit, shots=4096):
        from qiskit import transpile
        qc_copy = qc.copy()
        if qc_copy.num_clbits == 0:
            qc_copy = qc_copy.measure_all(inplace=False)
        tqc = transpile(qc_copy, self._backend_qasm)
        job = self._backend_qasm.run(tqc, noise_model=self.noise_model, shots=shots)
        return job.result().get_counts()
 
    def simulate_classical_statevector(self, qc: QuantumCircuit):
        num_qubits = qc.num_qubits
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1.0
        for instr, qargs, _ in qc.data:
            qubit_indices = [qc.find_bit(q).index for q in qargs]
            assert len(set(qubit_indices)) == len(qubit_indices), "Duplicate target qubits"
            assert all(0 <= t < num_qubits for t in qubit_indices), "Target qubit out of range"
            if hasattr(instr, "to_matrix"):
                mat = instr.to_matrix()
            else:
                name = instr.name.lower()
                if name == 'h':
                    mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                elif name == 'x':
                    mat = np.array([[0, 1], [1, 0]])
                elif name == 'z':
                    mat = np.array([[1, 0], [0, -1]])
                elif name in ('cx', 'cnot'):
                    mat = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]
                    ])
                else:
                    raise ValueError(f"Unsupported gate without to_matrix(): {instr.name}")
            state = apply_gate_statevector(state, mat, qubit_indices, num_qubits)
        return state
 
    @staticmethod
    def fidelity(state_a, state_b):
        na = np.linalg.norm(state_a)
        nb = np.linalg.norm(state_b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        inner = np.vdot(state_a, state_b) / (na * nb)
        return float(np.abs(inner) ** 2)
 
    def benchmark(self, num_qubits=3, depth=6, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        qc = random_circuit(num_qubits, depth)
        t0 = time.time()
        qstate = self.simulate_qiskit_statevector(qc)
        qt = time.time() - t0
        t0 = time.time()
        cstate = self.simulate_classical_statevector(qc)
        ct = time.time() - t0
        fid = self.fidelity(cstate, qstate)
        df = pd.DataFrame([{
            "Qubits": num_qubits,
            "Depth": depth,
            "Qiskit Time (s)": qt,
            "Classical Time (s)": ct,
            "Fidelity": fid
        }])
        return df, qc, qstate, cstate
 
 
if __name__ == "__main__":
    fw = QuantumFramework(noise_level=0.02)
    df, qc, qstate, cstate = fw.benchmark(num_qubits=2, depth=4, seed=42)
    print("Random Circuit:\n")
    print(qc)
    print("\nQiskit (ideal) statevector:\n", np.round(qstate, 5))
    print("\nClassical (custom) statevector:\n", np.round(cstate, 5))
    print("\nBenchmark:\n", df.to_string(index=False))
    noisy_counts = fw.simulate_qiskit_noisy_counts(qc, shots=2048)
    print("\nNoisy counts (qasm simulator with noise):\n", noisy_counts)
