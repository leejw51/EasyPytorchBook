#!/usr/bin/env python3
"""Recursively convert all Markdown files to PDF."""

import os
import subprocess
import sys
from pathlib import Path


def find_markdown_files(root_dir: str) -> list[Path]:
    """Recursively find all .md files under root_dir."""
    return sorted(Path(root_dir).rglob("*.md"))


def convert_md_to_pdf(md_path: Path) -> bool:
    """Convert a single Markdown file to PDF using pandoc."""
    pdf_path = md_path.with_suffix(".pdf")
    print(f"Converting: {md_path} -> {pdf_path}")
    try:
        subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-o",
                str(pdf_path),
                "--pdf-engine=xelatex",
                "-V",
                "geometry:margin=1in",
                "-V",
                "CJKmainfont=Arial Unicode MS",
                "-V",
                "mainfont=Helvetica Neue",
                "-V",
                "monofont=Menlo",
                "--syntax-highlighting=tango",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  OK: {pdf_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {md_path}")
        if e.stderr:
            print(f"  Error: {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        print("Error: pandoc not found. Install it with: brew install pandoc")
        print("Also need xelatex: brew install --cask mactex")
        sys.exit(1)


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    md_files = find_markdown_files(root_dir)

    if not md_files:
        print(f"No Markdown files found in {root_dir}")
        return

    print(f"Found {len(md_files)} Markdown files\n")

    success = 0
    failed = 0
    for md_file in md_files:
        if convert_md_to_pdf(md_file):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} converted, {failed} failed")


if __name__ == "__main__":
    main()
