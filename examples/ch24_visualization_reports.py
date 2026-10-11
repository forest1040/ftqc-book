"""Chapter 24: generate a real lattice-surgery mapping and timeline report."""
from pathlib import Path
from tempfile import TemporaryDirectory


def main(output_dir=None):
    print("=== Chapter 24 Example: Visualization & Reports ===")
    from ftqc.viz.ftqc_integrated_report import build_report_html

    def generate(directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "chapter24_ls_cnot_report.html"
        html = build_report_html("cnot", p=0.0, title="Chapter 24: LS CNOT")
        report_path.write_text(html, encoding="utf-8")
        assert "Lattice-surgery layout" in html
        assert "tick" in html.lower()
        return report_path

    if output_dir is not None:
        report_path = generate(output_dir)
        print(f"Generated {report_path}")
        return report_path

    with TemporaryDirectory(prefix="ftqc-ch24-") as directory:
        report_path = generate(directory)
        print(f"Generated and checked {report_path.name}")
        return report_path

if __name__ == "__main__":
    main()
