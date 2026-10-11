"""
ch18_shell_api.py - Stage 02 Example: Circuit Level Shell API & Handoff
"""
from ftqc.ls_handoff_shells import build_ls_s_shell

def main():
    print("=== Chapter 18 Example: Circuit Level Shell API ===")
    shell = build_ls_s_shell(p=0.001, merge_rounds=2)
    print(f"Shell circuit detector count: {shell.circuit.num_detectors}")

if __name__ == "__main__":
    main()
