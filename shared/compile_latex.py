"""
Compile D-HNSW paper to PDF using Modal cloud with full texlive.
"""
import modal
import os

app = modal.App("latex-compile")

latex_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "texlive-latex-base",
        "texlive-latex-recommended",
        "texlive-latex-extra",
        "texlive-fonts-recommended",
        "texlive-bibtex-extra",
        "texlive-science",
        "texlive-publishers",
        "cm-super",
        "lmodern",
    )
)

volume = modal.Volume.from_name("latex-build", create_if_missing=True)


@app.function(
    image=latex_image,
    volumes={"/workspace": volume},
    cpu=2,
    memory=4096,
    timeout=300,
)
def compile_paper(tex_content: str, bib_content: str, figures: dict):
    import subprocess
    import os

    build_dir = "/workspace/build"
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(f"{build_dir}/figures", exist_ok=True)

    # Write main.tex
    with open(f"{build_dir}/main.tex", "w") as f:
        f.write(tex_content)

    # Write references.bib
    with open(f"{build_dir}/references.bib", "w") as f:
        f.write(bib_content)

    # Write figure files
    for fname, data in figures.items():
        fpath = f"{build_dir}/figures/{fname}"
        with open(fpath, "wb") as f:
            f.write(data)
        print(f"  Wrote figure: {fname} ({len(data)} bytes)")

    # Compile: pdflatex -> bibtex -> pdflatex -> pdflatex
    os.chdir(build_dir)

    steps = [
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ]

    for i, cmd in enumerate(steps):
        print(f"\n{'='*60}")
        print(f"Step {i+1}: {' '.join(cmd)}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Print last 30 lines of output
        output = result.stdout + result.stderr
        output_lines = output.strip().split('\n')
        for line in output_lines[-30:]:
            print(line)
        if result.returncode != 0 and i == 1:
            # bibtex warnings are OK
            print(f"  (bibtex returned {result.returncode} — warnings OK)")
        elif result.returncode != 0 and i > 1:
            print(f"  WARNING: {cmd[0]} returned {result.returncode}")

    # Check if PDF was generated
    pdf_path = f"{build_dir}/main.pdf"
    if os.path.exists(pdf_path):
        pdf_size = os.path.getsize(pdf_path)
        print(f"\n✅ PDF generated: {pdf_size} bytes ({pdf_size/1024:.1f} KB)")

        # Read and return PDF
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        # Also check page count
        try:
            result = subprocess.run(
                ["pdfinfo", pdf_path], capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if 'Pages' in line:
                    print(f"  {line.strip()}")
        except:
            pass

        volume.commit()
        return pdf_data
    else:
        print("\n❌ PDF not generated!")
        # Print log for debugging
        if os.path.exists(f"{build_dir}/main.log"):
            with open(f"{build_dir}/main.log") as f:
                log = f.read()
            # Find errors
            for line in log.split('\n'):
                if '!' in line or 'Error' in line:
                    print(f"  ERROR: {line}")
        volume.commit()
        return None


@app.local_entrypoint()
def main():
    import os

    project_root = os.environ.get(
        "PROJECT_ROOT",
        "/home/orchestra/projects/80dcab4f-994f-46b3-a5ba-0333c988cd49"
    )
    paper_dir = f"{project_root}/extracted_workspace/workspace/agent_elsa_329d02a1_workdir"

    # Read main.tex
    with open(f"{paper_dir}/main.tex", "r") as f:
        tex_content = f.read()
    print(f"Read main.tex: {len(tex_content)} chars")

    # Read references.bib
    with open(f"{paper_dir}/references.bib", "r") as f:
        bib_content = f.read()
    print(f"Read references.bib: {len(bib_content)} chars")

    # Read all figure files
    figures = {}
    fig_dir = f"{paper_dir}/figures"
    for fname in os.listdir(fig_dir):
        if fname.endswith(('.pdf', '.png', '.jpg')):
            with open(f"{fig_dir}/{fname}", "rb") as f:
                figures[fname] = f.read()
            print(f"Read figure: {fname} ({len(figures[fname])} bytes)")

    print(f"\nTotal figures: {len(figures)}")
    print("Sending to Modal for compilation...")

    # Compile
    pdf_data = compile_paper.remote(tex_content, bib_content, figures)

    if pdf_data:
        # Save PDF locally
        output_path = f"{project_root}/shared/D_HNSW_IEEE_TKDE.pdf"
        with open(output_path, "wb") as f:
            f.write(pdf_data)
        print(f"\n✅ PDF saved to: {output_path} ({len(pdf_data)} bytes)")

        # Also save to agent workdir
        output_path2 = f"{project_root}/agent_workspace_extractor_and_analyzer_2e0e1b73_workdir/D_HNSW_IEEE_TKDE.pdf"
        with open(output_path2, "wb") as f:
            f.write(pdf_data)
        print(f"✅ PDF also saved to: {output_path2}")
    else:
        print("\n❌ Compilation failed!")
