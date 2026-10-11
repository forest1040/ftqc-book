"""ch15_hook_error.py - Stage 02 Example: Hook Error Propagation"""
from ftqc.surface_patch import SurfacePatch

def main():
    print("=== Chapter 15 Example: Hook Error Analysis ===")
    patch = SurfacePatch(distance=3)
    err, method = patch.decode(patch.memory_circuit(rounds=2, p=0.001), shots=100)
    print(f"Logical error rate: {err:.4f} via {method}")

if __name__ == '__main__':
    main()
