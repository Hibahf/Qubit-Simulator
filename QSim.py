import numpy as np
import pandas as pd
import random
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import HGate, XGate, ZGate, CXGate

# Linear-algebra helpers
def kron_n(matrices):
    """Kronecker product of a list of matrices in order."""
    res = np.array([[1.0]])
    for m in matrices:
        res = np.kron(res, m)
    return res

def apply_gate_statevector(state, gate_matrix, targets, num_qubits):
    """
    Apply a gate (matrix) acting on `len(targets)` qubits to the full `state`
    vector of dimension 2**num_qubits. This uses tensor reshapes and axis
    permutation to place target qubits in front, apply the gate, then undo.
    - state: (2**n,) complex numpy array
    - gate_matrix: (2**k, 2**k) complex numpy array where k = len(targets)
    - targets: list/iterable of distinct qubit indices (0 = least significant or user convention)
    - num_qubits: total number of qubits
    Convention used: qubit index 0 corresponds to the leftmost tensor axis in reshape:
      state_tensor.shape == (2,)*num_qubits
    """
    k = len(targets)
    if k == 0:
        return state
    if k == num_qubits:
        return gate_matrix @ state

    # reshape into tensor (axis 0..num_qubits-1)
    state_tensor = state.reshape([2] * num_qubits)

    # Build permutation: bring targets to front in the same order they appear in `targets`
    remaining = [i for i in range(num_qubits) if i not in targets]
    perm = list(targets) + remaining
    inv_perm = np.argsort(perm)

    permuted = np.transpose(state_tensor, axes=perm)

    dim_left = 2 ** k
    dim_right = 2 ** (num_qubits - k)
    reshaped = permuted.reshape((dim_left, dim_right))

    after = gate_matrix @ reshaped  # shape (dim_left, dim_right)

    after_tensor = after.reshape([2] * num_qubits)
    final = np.transpose(after_tensor, axes=inv_perm)
    return final.reshape(2 ** num_qubits)


# Random circuit helpers
def random_gate(num_qubits):
    """Return a gate object (Qiskit gate instance) and target list."""
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


# Framework class
class QuantumFramework:
    def __init__(self, noise_level=0.0):
        """
        noise_level: base depolarizing error probability (float). If 0.0, noise functions are effectively off.
        """
        self.noise_level = float(noise_level)
        self._backend_sv = Aer.get_backend("statevector_simulator")
        self._backend_qasm = Aer.get_backend("qasm_simulator")
        self.noise_model = self._make_noise_model() if self.noise_level > 0 else None

    def _make_noise_model(self):
        nm = NoiseModel()
        # simple depolarizing on 1q and 2q gates
        error_1q = depolarizing_error(self.noise_level, 1)
        error_2q = depolarizing_error(min(1.0, self.noise_level * 2.0), 2)
        nm.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'z'])
        nm.add_all_qubit_quantum_error(error_2q, ['cx'])
        return nm

    # Qiskit (ideal) sim
    def simulate_qiskit_statevector(self, qc: QuantumCircuit):
        """Return Qiskit's ideal statevector (no noise)."""
        job = execute(qc, self._backend_sv)
        result = job.result()
        return result.get_statevector(qc)

    # Qiskit noisy (qasm) sim (returns counts)
    def simulate_qiskit_noisy_counts(self, qc: QuantumCircuit, shots=4096):
        """
        Run qasm_simulator with the configured noise model. Returns counts dict.
        Note: noisy simulation returns measurement counts, not a statevector.
        """
        qc_copy = qc.copy()
        # ensure measurement on all qubits
        if qc_copy.num_clbits == 0 or len(qc_copy.clbits) == 0:
            qc_copy = qc_copy.measure_all(inplace=False)

        job = execute(qc_copy, self._backend_qasm, noise_model=self.noise_model, shots=shots)
        result = job.result()
        return result.get_counts()

    # Classical statevector simulation (ideal)
    def simulate_classical_statevector(self, qc: QuantumCircuit):
        """
        Deterministic classical simulation matching Qiskit statevector evolution.
        Iterates through qc.data and applies the gate matrices to the statevector.
        Supports gates with `to_matrix()` or common named gates (h, x, z, cx).
        """
        num_qubits = qc.num_qubits
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1.0  # |0...0>

        for instr, qargs, _ in qc.data:
            qubit_indices = [q.index for q in qargs]

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
                elif name == 'cx' or name == 'cnot':
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

    # Utilities
    @staticmethod
    def fidelity(state_a, state_b):
        """State fidelity |<a|b>|^2 for pure states."""
        na = np.linalg.norm(state_a)
        nb = np.linalg.norm(state_b)
        if na == 0 or nb == 0:
            return 0.0
        inner = np.vdot(state_a, state_b) / (na * nb)
        return float(np.abs(inner) ** 2)

    # Benchmark helper
    def benchmark(self, num_qubits=3, depth=6, seed=None):
        """
        Generate a random circuit, run Qiskit statevector (ideal) and classical simulator,
        measure timing and fidelity.
        Returns: (DataFrame, qc, qstate, cstate)
        """
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

# Example usage
if __name__ == "__main__":
    fw = QuantumFramework(noise_level=0.02)  # noise_level>0 enables noise model for qasm runs
    df, qc, qstate, cstate = fw.benchmark(num_qubits=2, depth=4, seed=42)

    print("Random Circuit:\n")
    print(qc)
    print("\nQiskit (ideal) statevector:\n", np.round(qstate, 5))
    print("\nClassical (custom) statevector:\n", np.round(cstate, 5))
    print("\nBenchmark:\n", df.to_string(index=False))

    # If you want noisy counts (qasm) for the same circuit:
    noisy_counts = fw.simulate_qiskit_noisy_counts(qc, shots=2048)
    print("\nNoisy counts (example, qasm simulator with noise):\n", noisy_counts)
