"""
ch17_ls_cnot.py - Stage 02 Example: Fault-Tolerant Circuit Level LS-CNOT
"""
from ftqc.ls_cnot_ft import LSCnotFT

def main():
    print("=== Chapter 17 Example: Circuit Level LS-CNOT ===")
    cnot = LSCnotFT()
    print("LSCnotFT initialized successfully.")

if __name__ == "__main__":
    main()
