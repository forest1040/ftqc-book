"""ch21_core7_integration.py - Stage 03 Example: Core 7 Module Collaboration"""
from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

def main():
    print("=== Chapter 21 Example: Core 7 Module Collaboration ===")
    runtime = HybridFTQCRuntime(n_logical=2)
    print("HybridFTQCRuntime Core 7 ready.")

if __name__ == '__main__':
    main()
