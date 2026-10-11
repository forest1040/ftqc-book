"""ch07_mpp_syndrome.py - Stage 01 Example: Phenomenological MPP Syndrome Extraction"""
from ftqc.surface_patch import mpp_memory_logical_error

def main():
    print("=== Chapter 7 Example: MPP Syndrome Extraction ===")
    err = mpp_memory_logical_error(p=0.005, rounds=2, shots=100)
    print(f"MPP logical error rate (p=0.005, rounds=2): {err:.4f}")

if __name__ == '__main__':
    main()
