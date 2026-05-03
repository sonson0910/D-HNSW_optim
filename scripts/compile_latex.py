#!/usr/bin/env python3
"""
D-HNSW IEEE TKDE LaTeX Compiler
Compiles the manuscript to PDF with proper IEEE formatting.

Usage:
    python scripts/compile_latex.py [--clean] [--check-only]
"""
import os, sys, subprocess, shutil
from pathlib import Path

BASE = Path(__file__).parent.parent
SHARED = BASE / "shared"
TEX_FILE = SHARED / "main_tex_final_audited.tex"
OUT_DIR = SHARED / "output"

def check_latex():
    """Check if LaTeX is available."""
    for cmd in ["pdflatex", "xelatex", "lualatex"]:
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split("\n")[0] if result.stdout else "unknown"
                print(f"  Found: {cmd} ({version})")
                return cmd
        except FileNotFoundError:
            continue
    return None

def check_bibtex():
    """Check if BibTeX is available."""
    try:
        result = subprocess.run(["bibtex", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_figures():
    """Verify all referenced figures exist."""
    fig_dir = SHARED / "figures"
    required = [
        "fig1_recall_qps_pareto.pdf",
        "fig2_hnsw_vs_dhnsw_sift.pdf",
        "fig3_multi_dataset.pdf",
        "fig4_error_analysis.pdf",
        "fig5_determinism_heatmap.pdf",
        "fig6_ablation_study.pdf",
        "fig7_scalability.pdf",
        "fig8_overhead_breakdown.pdf",
        "fig9_latency_comparison.pdf",
        "fig10_qualitative_comparison.pdf",
        "fig12_real_i64f32_search.pdf",
        "fig13_two_phase_search.pdf",
        "figB1_distance_ablation.pdf",
        "figB2_build_scaling.pdf",
    ]
    missing = []
    for f in required:
        path = fig_dir / f
        if path.exists():
            print(f"  [OK] {f} ({path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [MISSING] {f}")
            missing.append(f)
    return missing

def compile_pdf(latex_cmd, tex_file, out_dir):
    """Compile LaTeX to PDF with multiple passes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    # Set TEXINPUTS to find figures
    fig_path = str(SHARED / "figures") + os.pathsep
    env["TEXINPUTS"] = str(SHARED) + os.pathsep + fig_path + os.pathsep + "." + os.pathsep

    common_args = [
        latex_cmd,
        f"-output-directory={out_dir}",
        "-interaction=nonstopmode",
        "-halt-on-error",
        str(tex_file),
    ]

    # Pass 1: Initial compilation
    print("\n[Pass 1/3] Initial compilation...")
    result = subprocess.run(common_args, capture_output=True, text=True, cwd=str(SHARED), env=env)
    if result.returncode != 0:
        print(f"  WARNING: Pass 1 had issues (rc={result.returncode})")
        # Show last 20 lines of log for debugging
        log_file = out_dir / tex_file.stem
        log_path = log_file.with_suffix(".log")
        if log_path.exists():
            lines = log_path.read_text(errors='replace').split('\n')
            error_lines = [l for l in lines if l.startswith('!') or 'Error' in l]
            if error_lines:
                print("  Errors found:")
                for l in error_lines[:10]:
                    print(f"    {l}")
    else:
        print("  [OK]")
    
    # Pass 2: BibTeX (if available and .bib exists)
    bib_file = SHARED / "references.bib"
    if bib_file.exists() and check_bibtex():
        print("[Pass 2/3] BibTeX processing...")
        aux_file = out_dir / tex_file.stem
        bib_result = subprocess.run(
            ["bibtex", str(aux_file.with_suffix(".aux"))],
            capture_output=True, text=True, cwd=str(SHARED), env=env
        )
        if bib_result.returncode == 0:
            print("  [OK]")
        else:
            print(f"  WARNING: BibTeX issues (rc={bib_result.returncode})")
    else:
        print("[Pass 2/3] Skipping BibTeX (no .bib or bibtex not found)")
    
    # Pass 3: Final compilation (resolve references)
    print("[Pass 3/3] Final compilation...")
    result = subprocess.run(common_args, capture_output=True, text=True, cwd=str(SHARED), env=env)
    
    pdf_path = out_dir / tex_file.with_suffix(".pdf").name
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  SUCCESS: {pdf_path.name} ({size_mb:.2f} MB)")
        print(f"  Location: {pdf_path}")
        print(f"{'='*60}")
        return pdf_path
    else:
        print(f"\n  FAILED: PDF not generated. Check {out_dir / tex_file.stem}.log")
        return None

def clean_output(out_dir):
    """Remove auxiliary files."""
    exts = ['.aux', '.log', '.bbl', '.blg', '.toc', '.out', '.lof', '.lot', '.fls', '.fdb_latexmk']
    for f in out_dir.glob('*'):
        if f.suffix in exts:
            f.unlink()
            print(f"  Removed {f.name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="D-HNSW LaTeX Compiler")
    parser.add_argument("--clean", action="store_true", help="Clean auxiliary files")
    parser.add_argument("--check-only", action="store_true", help="Only check prerequisites")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" D-HNSW IEEE TKDE Manuscript Compiler")
    print("=" * 60)
    
    if args.clean:
        print("\nCleaning output directory...")
        clean_output(OUT_DIR)
        return
    
    # Step 1: Check LaTeX
    print("\n[1/3] Checking LaTeX installation...")
    latex_cmd = check_latex()
    if not latex_cmd:
        print("  LaTeX not found!")
        print("  Install MiKTeX (Windows): https://miktex.org/download")
        print("  Or TeX Live: https://www.tug.org/texlive/")
        if args.check_only:
            return
        print("\n  Attempting to install MiKTeX via winget...")
        subprocess.run(["winget", "install", "MiKTeX.MiKTeX"], capture_output=True)
        latex_cmd = check_latex()
        if not latex_cmd:
            print("  FATAL: Cannot compile without LaTeX. Please install manually.")
            sys.exit(1)
    
    # Step 2: Check figures
    print("\n[2/3] Checking figures...")
    missing = check_figures()
    if missing:
        print(f"\n  {len(missing)} figure(s) missing. Run: python scripts/generate_figures.py")
        if args.check_only:
            return
    
    # Step 3: Check tex file
    print(f"\n[3/3] Checking manuscript: {TEX_FILE.name}")
    if TEX_FILE.exists():
        lines = TEX_FILE.read_text(errors='replace').count('\n')
        print(f"  [OK] {lines} lines, {TEX_FILE.stat().st_size / 1024:.1f} KB")
    else:
        print(f"  FATAL: {TEX_FILE} not found!")
        sys.exit(1)
    
    if args.check_only:
        print("\n  Prerequisites check complete.")
        return
    
    # Compile
    pdf = compile_pdf(latex_cmd, TEX_FILE, OUT_DIR)
    if pdf:
        # Copy to shared root for easy access
        final = SHARED / "D-HNSW_IEEE_TKDE.pdf"
        shutil.copy2(pdf, final)
        print(f"\n  Final PDF: {final}")

if __name__ == "__main__":
    main()
