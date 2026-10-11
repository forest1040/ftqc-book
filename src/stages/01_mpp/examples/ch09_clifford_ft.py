"""ch09_clifford_ft.py - Stage 01 Example: Transversal & Geometric Clifford FT"""
from ftqc.clifford_ft_full import CliffordFTCompiler

def main():
    print("=== Chapter 9 Example: Clifford FT Compiler ===")
    compiler = CliffordFTCompiler(n_logical=2)
    print("CliffordFTCompiler initialized successfully.")

if __name__ == '__main__':
    main()
