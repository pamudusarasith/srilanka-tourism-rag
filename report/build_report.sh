#!/usr/bin/env bash
# Compile report.tex to report.pdf.
#
# xelatex is run twice so the table of contents and cross-references settle.
# MacTeX installs to /Library/TeX/texbin, which is frequently absent from
# PATH, so that location is added explicitly if xelatex isn't already found.
#
# Run:  ./report/build_report.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/.build"
TEXBIN="/Library/TeX/texbin"

if ! command -v xelatex >/dev/null 2>&1; then
    if [ -x "$TEXBIN/xelatex" ]; then
        PATH="$TEXBIN:$PATH"
    else
        echo "xelatex not found (install MacTeX or TeX Live)" >&2
        exit 1
    fi
fi

mkdir -p "$BUILD"
cp "$HERE/report.tex" "$BUILD/report.tex"

cd "$BUILD"
for pass in 1 2; do
    if ! xelatex -interaction=nonstopmode -halt-on-error report.tex > /dev/null; then
        echo "xelatex failed on pass $pass:" >&2
        grep '^! ' report.log | head -10 >&2 || true
        exit 1
    fi
done

cp report.pdf "$HERE/report.pdf"
echo "Wrote $HERE/report.pdf"
