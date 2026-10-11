"""
ch08_lattice_surgery.py - Stage 01 Example: Phenomenological Lattice Surgery
"""
from ftqc.geom_lattice_surgery import GeomMerge

def main():
    print("=== Chapter 8 Example: Phenomenological Lattice Surgery ===")
    geom = GeomMerge(kind="ZZ")
    print(f"Merge kind: {geom.kind}, data qubits: {geom.n}")

if __name__ == "__main__":
    main()
