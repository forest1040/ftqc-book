"""
ch10_t_gate.py - Stage 01 Example: Magic State Distillation & Injection
"""
from ftqc.t_gate import build_15to1, distill

def main():
    print("=== Chapter 10 Example: 15-to-1 Magic State Distillation ===")
    HX, HZ = build_15to1()
    acc, err = distill(HX, HZ, p=0.001, shots=50)
    print(f"Distillation acceptance: {acc:.4f}, error: {err:.4f}")

if __name__ == "__main__":
    main()
