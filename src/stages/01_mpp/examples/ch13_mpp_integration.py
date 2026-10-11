"""ch13_mpp_integration.py - Stage 01 Example: Stage 01 Integrated Verification"""
from ftqc.surface_patch import SurfacePatch
from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

def main():
    print("=== Chapter 13 Example: Stage 01 Integrated Verification ===")
    patch = SurfacePatch(distance=3)
    runtime = HybridFTQCRuntime(n_logical=2)
    print(f"Stage 01 patch data qubits: {len(patch.data)}, runtime ready.")

if __name__ == '__main__':
    main()
