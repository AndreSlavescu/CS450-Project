#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TEX_FILE="report.tex"

# Generate acmart.cls if not present (requires acmart.dtx + acmart.ins)
if [ ! -f acmart.cls ]; then
    echo "Generating acmart.cls..."
    curl -sL "https://mirrors.ctan.org/macros/latex/contrib/acmart.zip" -o acmart.zip
    unzip -o acmart.zip "acmart/acmart.ins" "acmart/acmart.dtx" -d /tmp
    pdflatex -interaction=nonstopmode -output-directory=/tmp/acmart /tmp/acmart/acmart.ins > /dev/null
    cp /tmp/acmart/acmart.cls .
    rm -f acmart.zip
fi

pdflatex -interaction=nonstopmode "$TEX_FILE"
bibtex "${TEX_FILE%.tex}"
pdflatex -interaction=nonstopmode "$TEX_FILE"
pdflatex -interaction=nonstopmode "$TEX_FILE"

cp "${TEX_FILE%.tex}.pdf" final_report.pdf
echo "Done: final_report.pdf"
