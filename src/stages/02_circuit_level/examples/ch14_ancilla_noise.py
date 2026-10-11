"""ch14_ancilla_noise.py - Stage 02 Example: Ancilla Noise Model"""
from ftqc.surface_patch import SurfacePatch

def main():
    print("=== Chapter 14 Example: Ancilla Qubit Noise Model ===")
    patch = SurfacePatch(distance=3)
    circ = patch.memory_circuit(rounds=2, p=0.001)
    print(f"Circuit qubits: {circ.num_qubits}, detectors: {circ.num_detectors}")

if __name__ == '__main__':
    main()
