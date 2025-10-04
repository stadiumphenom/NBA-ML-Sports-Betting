#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
import shutil

NBA_KEYWORDS = [
    "nba", "nba.com", "nba-2024", "TEAM_NAME", "00_todays_scores.json",
    "leaguedashteamstats"
]

BAD_FILE_PATTERNS = [
    "nba", "NBA", "SbrOddsProvider", "Get_Data", "Get_Odds", "Fix_Odds", "UTC.csv"
]

EXCLUDE_DIRS = [".git", ".github", "__pycache__", "venv", "env", ".mypy_cache"]

TRASH_DIR = "trash"


def scan_file(filepath: Path):
    """Scan file for NBA leftovers and return flagged lines."""
    results = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                for kw in NBA_KEYWORDS:
                    if kw in line:
                        results.append((i, line.strip(), kw))
    except Exception as e:
        return [(0, f"[Error reading file: {e}]", None)]
    return results


def scan_repo(root: str = "src"):
    """Walk repo and scan for NBA leftovers."""
    report = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            path = Path(dirpath) / fname

            # Flag bad filenames
            for pattern in BAD_FILE_PATTERNS:
                if pattern in fname:
                    report[str(path)] = [(0, f"Filename contains {pattern}", pattern)]
                    break

            # Scan file contents
            findings = scan_file(path)
            if findings:
                report.setdefault(str(path), []).extend(findings)
    return report


def auto_fix(report):
    """Auto-fix detected junk (move files or replace keywords)."""
    Path(TRASH_DIR).mkdir(exist_ok=True)

    for file, findings in report.items():
        file_path = Path(file)

        # Delete/move bad files
        for line, content, kw in findings:
            if line == 0 and kw in BAD_FILE_PATTERNS:
                print(f"🗑 Moving bad file {file} to {TRASH_DIR}/")
                shutil.move(file, Path(TRASH_DIR) / file_path.name)
                break

        # Replace NBA keywords inside files
        if file_path.exists():
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            new_content = content
            for kw in NBA_KEYWORDS:
                if "nba" in kw.lower():
                    new_content = new_content.replace(kw, kw.replace("nba", "nfl"))

            if new_content != content:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"✏️ Replaced NBA keywords in {file}")


def main():
    parser = argparse.ArgumentParser(description="Scan repo for NBA leftovers and junk.")
    parser.add_argument("--root", default="src", help="Root directory to scan")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    args = parser.parse_args()

    report = scan_repo(args.root)

    if not report:
        print("✅ No NBA leftovers found. Repo looks clean!")
        return

    print("🚨 NBA leftovers / junk detected:\n")
    for file, findings in report.items():
        print(f"File: {file}")
        for line, content, kw in findings:
            if line == 0:
                print(f"  [!] {content}")
            else:
                print(f"  Line {line}: {content}")
        print("")

    if args.fix:
        print("⚡ Running auto-fix...")
        auto_fix(report)
        print("✅ Auto-fix completed. Check trash/ folder for removed files.")


if __name__ == "__main__":
    main()
