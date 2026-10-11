"""
ch16_circuit_merge.py - Stage 02 Example: Circuit Level Merge Operation
"""
from ftqc.circuit_level_merge import CircuitLevelMerge

def main():
    print("=== Chapter 16 Example: Circuit Level Merge ===")
    cm = CircuitLevelMerge(kind="ZZ")
    print(f"CircuitLevelMerge initialized for kind={cm.kind}")

if __name__ == "__main__":
    main()
