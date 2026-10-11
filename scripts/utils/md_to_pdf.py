#!/usr/bin/env python3
"""Convert a Markdown file to PDF with embedded images and block math.

Block math (``$$...$$``) is rendered to SVG with matplotlib mathtext;
inline ``$...$`` math is not rendered. Local images are embedded as base64
data URIs and the PDF is written with WeasyPrint (A4 landscape).

Usage:
    python scripts/utils/md_to_pdf.py notes.md
    python scripts/utils/md_to_pdf.py notes.md -o out/notes.pdf
"""

import argparse
import base64
import io
import mimetypes
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import markdown
from weasyprint import HTML


_CSS = """
body { font-family: sans-serif; margin: 30px; font-size: 10pt; line-height: 1.35; color: #222; }
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 3px; margin-top: 0; margin-bottom: 4px; }
h2 { font-size: 15pt; border-bottom: 1px solid #999; padding-bottom: 2px; margin-top: 18px; margin-bottom: 4px; page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 12pt; margin-top: 12px; margin-bottom: 2px; }
table { border-collapse: collapse; width: 100%; margin: 4px 0; font-size: 7.5pt; }
th, td { border: 1px solid #aaa; padding: 2px 4px; text-align: center; }
th { background: #e8e8e8; }
img { max-width: 100%; max-height: 85vh; height: auto; width: auto; display: block; margin: 4px auto; page-break-inside: avoid; }
img.math-block { margin: 6px auto; max-height: 60px; }
img.math-inline { display: inline; vertical-align: middle; height: 1.2em; margin: 0 2px; }
code { background: #f4f4f4; padding: 1px 3px; border-radius: 3px; font-size: 9pt; }
pre { background: #f4f4f4; padding: 6px; border-radius: 4px; overflow-x: auto; margin: 4px 0; }
p { margin: 3px 0; }
ul, ol { margin: 3px 0; padding-left: 20px; }
li { margin: 1px 0; }
hr { margin: 4px 0; }
@page { size: A4 landscape; margin: 1cm; }
"""

_math_cache: dict[str, str] = {}


def _render_math_svg(tex: str, display: bool = True) -> str:
    """Render a LaTeX expression to an SVG data URI using matplotlib mathtext.

    Results are memoised in ``_math_cache`` keyed by ``(tex, display)``.

    Args:
        tex: Math expression without surrounding ``$`` delimiters.
        display: If ``True`` use display sizing, otherwise inline sizing.

    Returns:
        str: ``data:image/svg+xml;base64,...`` URI.

    Raises:
        Exception: Propagates matplotlib errors for unparsable expressions.
    """
    key = (tex, display)
    if key in _math_cache:
        return _math_cache[key]
    fontsize = 14 if display else 11
    fig, ax = plt.subplots(figsize=(12 if display else 8, 1.2 if display else 0.5))
    ax.axis("off")
    ax.text(
        0.5, 0.5, f"${tex}$",
        fontsize=fontsize, ha="center", va="center",
        transform=ax.transAxes, math_fontfamily="cm",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    uri = f"data:image/svg+xml;base64,{b64}"
    _math_cache[key] = uri
    return uri


def _render_math_blocks(text: str) -> str:
    """Replace ``$$...$$`` block math with text placeholders before markdown.

    Handles both multi-line blocks (``$$`` on their own lines) and
    single-line ``$$...$$``. Each match becomes a paragraph containing a
    ``MATHBLOCK<n>END`` tag.

    Args:
        text: Raw Markdown source.

    Returns:
        tuple[str, dict]: The rewritten text and a mapping from placeholder
        tag to the stripped TeX source. (The ``-> str`` annotation is
        inaccurate.)
    """
    counter = {"n": 0}
    placeholders = {}

    def _block_repl(m: re.Match) -> str:
        """Record the matched TeX and return its placeholder paragraph.

        Args:
            m: Regex match whose group 1 is the TeX source.

        Returns:
            str: Placeholder tag surrounded by blank lines.
        """
        tex = m.group(1).strip()
        tag = f"MATHBLOCK{counter['n']}END"
        counter["n"] += 1
        placeholders[tag] = tex
        return f"\n\n{tag}\n\n"

    # Block math: $$ on own line ... $$ on own line
    text = re.sub(
        r'^\$\$\s*\n(.*?)\n\s*\$\$',
        _block_repl, text, flags=re.MULTILINE | re.DOTALL,
    )
    # Single-line block math: $$...$$
    text = re.sub(
        r'\$\$(.+?)\$\$',
        _block_repl, text,
    )
    return text, placeholders


def _resolve_math_placeholders(html: str, placeholders: dict) -> str:
    """Replace MATHBLOCK placeholders in HTML with rendered SVG images.

    A placeholder wrapped in ``<p>`` is replaced as a whole; render failures
    are replaced with a red ``<pre>`` error block instead of raising.

    Args:
        html: HTML produced by ``markdown.markdown``.
        placeholders: Mapping from placeholder tag to TeX source.

    Returns:
        str: HTML with placeholders substituted.
    """
    for tag, tex in placeholders.items():
        try:
            uri = _render_math_svg(tex, display=True)
            img = f'<img class="math-block" src="{uri}" alt="{tex[:60]}" />'
        except Exception as e:
            img = f'<pre style="color:red;">Math error: {e}\n{tex}</pre>'
        html = html.replace(f"<p>{tag}</p>", img)
        html = html.replace(tag, img)
    return html


def _embed_images(html: str, base_dir: Path) -> str:
    """Inline local ``<img>`` sources as base64 data URIs.

    ``http(s)://`` and ``data:`` sources, and paths that do not resolve to an
    existing file, are left unchanged.

    Args:
        html: HTML document body.
        base_dir: Directory that relative ``src`` paths are resolved against.

    Returns:
        str: HTML with local image sources embedded.
    """
    def _replace(m: re.Match) -> str:
        """Return the ``src`` attribute rewritten to a data URI when possible.

        Args:
            m: Regex match with groups (prefix, src, closing quote).

        Returns:
            str: Rewritten attribute text, or the original match.
        """
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
    """Convert a Markdown file to PDF.

    Args:
        md_path: Input Markdown file.
        pdf_path: Output PDF path; defaults to ``md_path`` with a ``.pdf``
            suffix.

    Returns:
        Path: Resolved path of the written PDF.
    """
    md_path = md_path.resolve()
    if pdf_path is None:
        pdf_path = md_path.with_suffix(".pdf")
    else:
        pdf_path = pdf_path.resolve()

    text = md_path.read_text(encoding="utf-8")

    # Extract and placeholder block math before markdown processing
    text, math_placeholders = _render_math_blocks(text)

    extensions = ["tables", "fenced_code", "toc"]

    body = markdown.markdown(text, extensions=extensions)
    body = _resolve_math_placeholders(body, math_placeholders)
    body = _embed_images(body, md_path.parent)

    full_html = f"<html><head><style>{_CSS}</style></head><body>{body}</body></html>"
    HTML(string=full_html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    return pdf_path


def main():
    """Parse CLI args and convert the input Markdown file to PDF.

    Exits with status 1 if the input file does not exist.
    """
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
