#!/usr/bin/env python3
"""Convert a Markdown file to PDF with embedded images."""

import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML


_CSS = """
body { font-family: sans-serif; margin: 40px; font-size: 11pt; line-height: 1.5; color: #222; }
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 16pt; border-bottom: 1px solid #999; padding-bottom: 3px; margin-top: 28px; page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 13pt; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 8pt; }
th, td { border: 1px solid #aaa; padding: 3px 5px; text-align: center; }
th { background: #e8e8e8; }
img { max-width: 100%; max-height: 90vh; height: auto; width: auto; display: block; margin: 10px auto; page-break-inside: avoid; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 10pt; }
pre { background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }
@page { size: A4 landscape; margin: 1cm; }
"""


def _embed_images(html: str, base_dir: Path) -> str:
    """Replace relative image src with base64 data URIs so WeasyPrint can render them."""
    def _replace(m: re.Match) -> str:
        prefix, src, suffix = m.group(1), m.group(2), m.group(3)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.is_file():
            return m.group(0)
        mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
        data = base64.b64encode(img_path.read_bytes()).decode()
        return f'{prefix}data:{mime};base64,{data}{suffix}'

    return re.sub(r'(<img\s[^>]*src=["\'])([^"\']+)(["\'])', _replace, html)


def convert(md_path: Path, pdf_path: Path | None = None) -> Path:
    md_path = md_path.resolve()
    if pdf_path is None:
        pdf_path = md_path.with_suffix(".pdf")
    else:
        pdf_path = pdf_path.resolve()

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    body = _embed_images(body, md_path.parent)

    full_html = f"<html><head><style>{_CSS}</style></head><body>{body}</body></html>"
    HTML(string=full_html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF")
    parser.add_argument("input", type=Path, help="Path to the Markdown (.md) file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PDF path (default: same name as input with .pdf)")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    out = convert(args.input, args.output)
    print(f"PDF written to {out}")


if __name__ == "__main__":
    main()
