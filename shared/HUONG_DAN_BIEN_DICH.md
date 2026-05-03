# Hướng Dẫn Biên Dịch Paper D-HNSW IEEE TKDE

## Cấu trúc thư mục cần có

```
paper/
├── main.tex              ← File LaTeX chính
├── references.bib        ← Tài liệu tham khảo
└── figures/              ← Thư mục hình ảnh
    ├── fig1_recall_qps_pareto.pdf
    ├── fig2_hnsw_vs_dhnsw_sift.pdf
    ├── fig3_multi_dataset.pdf
    ├── fig4_error_analysis.pdf
    ├── fig5_determinism_heatmap.pdf
    ├── fig6_ablation_study.pdf
    ├── fig7_scalability.pdf
    ├── fig8_overhead_breakdown.pdf
    ├── fig9_latency_comparison.pdf
    ├── fig10_qualitative_comparison.pdf
    ├── fig12_real_i64f32_search.png
    ├── fig13_two_phase_search.png
    └── fig15_determinism_verification.png
```

## Cách 1: Biên dịch trên máy local (Khuyến nghị)

### Bước 1: Cài đặt TeX Live

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
  texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra \
  texlive-science texlive-publishers cm-super lmodern
```

**macOS (Homebrew):**
```bash
brew install --cask mactex
```

**Windows:**
- Tải MiKTeX từ https://miktex.org/download
- Hoặc TeX Live từ https://tug.org/texlive/

### Bước 2: Biên dịch (4 bước)

```bash
cd paper/

# Bước 1: pdflatex lần 1 (tạo .aux)
pdflatex -interaction=nonstopmode main.tex

# Bước 2: bibtex (xử lý references)
bibtex main

# Bước 3: pdflatex lần 2 (cập nhật references)
pdflatex -interaction=nonstopmode main.tex

# Bước 4: pdflatex lần 3 (hoàn thiện cross-references)
pdflatex -interaction=nonstopmode main.tex
```

### Bước 3: Kiểm tra kết quả

```bash
ls -la main.pdf
# Kỳ vọng: ~1MB, 15 trang
```

## Cách 2: Script tự động (1 lệnh)

Tạo file `build.sh`:

```bash
#!/bin/bash
set -e

echo "=== Building D-HNSW IEEE TKDE Paper ==="

cd "$(dirname "$0")"

echo "[1/4] pdflatex (pass 1)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

echo "[2/4] bibtex..."
bibtex main > /dev/null 2>&1 || true

echo "[3/4] pdflatex (pass 2)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

echo "[4/4] pdflatex (pass 3)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

echo ""
echo "✅ Done! Output: main.pdf"
ls -lh main.pdf
```

Chạy:
```bash
chmod +x build.sh
./build.sh
```

## Cách 3: Overleaf (Online, không cần cài đặt)

1. Truy cập https://www.overleaf.com
2. **New Project** → **Upload Project**
3. Upload toàn bộ thư mục `paper/` (main.tex, references.bib, figures/)
4. Overleaf sẽ tự động biên dịch
5. Nhấn **Recompile** để xem PDF

**Lưu ý Overleaf:**
- Compiler: chọn **pdfLaTeX**
- Main document: chọn **main.tex**

## Cách 4: Docker (Portable)

```bash
docker run --rm -v "$(pwd)/paper:/workdir" -w /workdir \
  texlive/texlive:latest \
  sh -c "pdflatex -interaction=nonstopmode main.tex && \
         bibtex main && \
         pdflatex -interaction=nonstopmode main.tex && \
         pdflatex -interaction=nonstopmode main.tex"
```

## Dọn dẹp file tạm

```bash
rm -f main.aux main.bbl main.blg main.log main.out main.toc main.lof main.lot
```

## Thông tin kỹ thuật

| Thuộc tính | Giá trị |
|-----------|---------|
| Document class | `IEEEtran` (journal mode, 10pt) |
| Font | Times New Roman (chuẩn IEEE) |
| Trang | 15 |
| Kích thước PDF | ~1 MB |
| Compiler | pdfLaTeX |
| BibTeX style | IEEEtran |

## Lưu ý quan trọng

1. **Phải chạy 4 bước** (pdflatex → bibtex → pdflatex → pdflatex) để references và cross-references hiển thị đúng
2. **Không dùng XeLaTeX hay LuaLaTeX** — IEEEtran được thiết kế cho pdfLaTeX
3. Nếu thiếu package, cài thêm: `sudo apt-get install texlive-full` (cài đầy đủ ~5GB)
4. Hình PNG (fig12, fig13, fig15) chỉ tương thích pdfLaTeX (không dùng được với latex+dvips)


---

<a href="https://www.orchestra-research.com/"><img src="https://img.shields.io/badge/Orchestra-Research-6C3FC5.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+" alt="Orchestra Research"></a>
