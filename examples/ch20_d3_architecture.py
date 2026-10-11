"""ch20_d3_architecture.py - Stage 03 Example: d=3 Integrated Architecture"""
from ftqc.ftqc_tsim_backend import TsimStateBackend, validate_distance_d3

def main():
    print("=== Chapter 20 Example: d=3 Integrated Architecture ===")
    validate_distance_d3(3)
    backend = TsimStateBackend(n=2, distance=3)
    print(f"Backend initialized for d={backend.d}, n={backend.n}")

if __name__ == '__main__':
    main()
