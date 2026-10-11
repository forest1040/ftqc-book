"""
ch19_circuit_integration.py - Stage 02 Example: Circuit Level Integrated Simulation
"""
from ftqc.ls_cnot_ft import LSCnotFT
from ftqc.circuit_level_merge import CircuitLevelMerge

def main():
    print("=== Chapter 19 Example: Circuit Level Integrated Simulation ===")
    cnot = LSCnotFT()
    merge = CircuitLevelMerge("ZZ")
    print(f"LSCnotFT and CircuitLevelMerge(ZZ) initialized successfully.")

if __name__ == "__main__":
    main()
