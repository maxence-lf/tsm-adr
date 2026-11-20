#!/bin/sh
# Build docs/report.pdf and docs/technical_report.pdf with pandoc and typst.
# Run from the repository root after make_report_tables.py. Requires pandoc and typst.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/docs"  # figures are referenced as ../figures/
for doc in report technical_report; do
    pandoc "$doc.md" -f gfm+implicit_figures -o "$doc.pdf" --pdf-engine=typst \
        -V papersize=a4 -V margin.x=2cm -V margin.y=2cm -V fontsize=10pt \
        -H "$ROOT/scripts/pdf_style.typ"
    echo "docs/$doc.pdf"
done
