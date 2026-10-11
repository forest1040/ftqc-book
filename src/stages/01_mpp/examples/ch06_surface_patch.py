"""
ch06_surface_patch.py - Stage 01 Example: Surface Code Patch & Stabilizers
"""
from ftqc.surface_patch import SurfacePatch

def main():
    print("=== Chapter 6 Example: Surface Patch ===")
    patch = SurfacePatch(distance=3)
    print(f"Data qubits count: {len(patch.data)}")
    print(f"Ancilla qubits count: {len(patch.anc)}")
    print(f"Stabilizers count: {len(patch.stabilizers)}")
    print(f"XL operators: {patch.XL}")
    print(f"ZL operators: {patch.ZL}")

if __name__ == "__main__":
    main()
