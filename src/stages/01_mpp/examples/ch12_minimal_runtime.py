"""ch12_minimal_runtime.py - Stage 01 Example: Minimal Hybrid Runtime"""
from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

def main():
    print("=== Chapter 12 Example: Minimal Hybrid Runtime ===")
    runtime = HybridFTQCRuntime(n_logical=2)
    print("HybridFTQCRuntime initialized successfully.")

if __name__ == '__main__':
    main()
