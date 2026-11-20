"""Consistency checks of the public documents (README.md, docs/report.md, docs/technical_report.md).

1. Numbers: every number quoted in docs/report.md and README.md must appear in
   docs/technical_report.md or in a generated table of docs/tables/. Code spans, code
   blocks, URLs and link targets are ignored. Numbers are compared after removing a
   leading sign, thousands separators and a trailing percent sign, so "-0.25" in the
   short report matches "0.25" in the technical report; this is a presence check, not
   a check of meaning.
2. Links: every relative link or image target exists, and every internal anchor
   (#section) matches a heading of the target document.
3. Hygiene: no absolute local path and no reference to the local folders that are
   never published.

Usage: python scripts/check_report_numbers.py   (exit code 1 if a check fails)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCES = [ROOT / "README.md", DOCS / "report.md"]
REFERENCE = [DOCS / "technical_report.md"]
PUBLIC = [ROOT / "README.md", DOCS / "report.md", DOCS / "technical_report.md"]
FORBIDDEN = [r"/Users/", r"/private/", r"/home/", r"notes/", r"interim_report", r"(?<![\w/])report/"]

NUMBER = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?%?")


def strip_markup(text: str) -> str:
    """Remove code, URLs, link targets and HTML comments, keeping visible text."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^(    |\t).*$", " ", text, flags=re.M)  # indented code blocks
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images: keep alt text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links: keep link text
    text = re.sub(r"https?://\S+", " ", text)
    return text


def normalise(token: str) -> str:
    token = token.lstrip("+-−").rstrip("%").replace(",", "")
    return token.rstrip(".")


def numbers(text: str) -> list[tuple[str, str]]:
    """(normalised number, context) for every number in the visible text."""
    clean = strip_markup(text)
    out = []
    for m in NUMBER.finditer(clean):
        context = clean[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")
        out.append((normalise(m.group()), context))
    return out


def reference_numbers() -> set[str]:
    texts = [p.read_text() for p in REFERENCE] + [p.read_text() for p in sorted((DOCS / "tables").glob("*.md"))]
    return {n for t in texts for n, _ in numbers(t)}


def check_numbers() -> list[str]:
    ref = reference_numbers()
    problems = []
    for path in SOURCES:
        for n, ctx in numbers(path.read_text()):
            if n not in ref:
                problems.append(f"{path.relative_to(ROOT)}: {n} not found in the technical report or tables "
                                f"(...{ctx.strip()}...)")
    return problems


def slug(heading: str) -> str:
    """GitHub-style anchor of a Markdown heading."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", " ", path.read_text(), flags=re.S)
    return {slug(h) for h in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M)}


def check_links() -> list[str]:
    problems = []
    for path in PUBLIC:
        text = re.sub(r"```.*?```", " ", path.read_text(), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"(https?:|mailto:)", target):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (path.parent / file_part).resolve() if file_part else path
            if not dest.exists():
                problems.append(f"{path.relative_to(ROOT)}: missing link target {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                problems.append(f"{path.relative_to(ROOT)}: missing anchor {target}")
    return problems


def check_hygiene() -> list[str]:
    problems = []
    for path in PUBLIC + sorted((DOCS / "tables").glob("*.md")):
        text = path.read_text()
        for pattern in FORBIDDEN:
            for m in re.finditer(pattern, text):
                problems.append(f"{path.relative_to(ROOT)}: forbidden reference '{m.group()}' "
                                f"(...{text[max(0, m.start() - 40):m.end() + 20]}...)")
    return problems


def main() -> int:
    results = {"numbers": check_numbers(), "links": check_links(), "hygiene": check_hygiene()}
    for name, problems in results.items():
        print(f"{name}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
        for p in problems:
            print("  " + p)
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
