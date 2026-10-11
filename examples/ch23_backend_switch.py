"""ch23_backend_switch.py - Stage 03 Example: Backend Automatic Switching"""
from ftqc.ftqc_tsim_backend import TsimStateBackend

def main():
    print("=== Chapter 23 Example: Backend Automatic Switching ===")
    backend = TsimStateBackend(n=2, distance=3)
    print(f"TsimStateBackend: name={backend.name}")

if __name__ == '__main__':
    main()
