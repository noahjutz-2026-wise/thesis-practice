#!/usr/bin/env python3
"""
wordcount_history.py

A dependency-free Python script that:
1. Iterates over git commits in the thesis repository.
2. Computes the word count for each commit using the Typst -> Pandoc
   compilation pipeline (as described in the README).
3. Caches results in a JSON file to avoid re-compiling existing commits.
4. Exports a JSON array of entries:
      [{"hash": ..., "timestamp": ..., "wc": ...}, ...]
5. Renders a standalone vector SVG plot of the word count over time.
"""

import argparse
import concurrent.futures
import datetime
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile


def run_git(args, repo_dir):
    res = subprocess.run(
        ["git", "-C", repo_dir] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def get_commit_list(repo_dir, rev_range="HEAD", reverse=True):
    """Retrieve list of commits (hash, unix_timestamp, iso_timestamp, subject)."""
    args = ["log", "--format=%H|%ct|%cI|%s"]
    if reverse:
        args.append("--reverse")
    args.append(rev_range)
    lines = run_git(args, repo_dir).splitlines()
    commits = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) >= 3:
            commits.append({
                "hash": parts[0],
                "timestamp_unix": int(parts[1]),
                "timestamp": parts[2],
                "subject": parts[3] if len(parts) > 3 else "",
            })
    return commits


def compute_commit_wc(repo_dir, commit_hash):
    """
    Computes word count for a given commit according to README instructions:
    typst compile ... --features html --format html ... src/main.typ src/main.html &&
    pandoc src/main.html -t plain --wrap none -o main.txt
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract archive of commit at this commit
        archive_proc = subprocess.run(
            ["git", "-C", repo_dir, "archive", commit_hash],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if archive_proc.returncode != 0:
            return None

        tar_proc = subprocess.run(
            ["tar", "-x", "-C", tmpdir],
            input=archive_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tar_proc.returncode != 0:
            return None

        src_main = os.path.join(tmpdir, "src", "main.typ")
        if not os.path.isfile(src_main):
            # No typst document at this commit
            return 0

        # Check if submodule vendor/literature exists at this commit
        sub_info = subprocess.run(
            ["git", "-C", repo_dir, "ls-tree", commit_hash, "vendor/literature"],
            capture_output=True,
            text=True,
        ).stdout.split()

        sub_copied = False
        lit_repo = os.path.join(repo_dir, "vendor", "literature")
        if len(sub_info) >= 3 and os.path.isdir(lit_repo):
            sub_commit = sub_info[2]
            sub_dir = os.path.join(tmpdir, "vendor", "literature")
            os.makedirs(sub_dir, exist_ok=True)
            try:
                sub_arch = subprocess.run(
                    ["git", "-C", lit_repo, "archive", sub_commit],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=True,
                ).stdout
                subprocess.run(
                    ["tar", "-x", "-C", sub_dir],
                    input=sub_arch,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                sub_copied = True
            except Exception:
                pass

        # Fallback to local working copy of vendor/literature if needed
        if not sub_copied and os.path.isdir(lit_repo):
            sub_dir = os.path.join(tmpdir, "vendor", "literature")
            os.makedirs(os.path.dirname(sub_dir), exist_ok=True)
            shutil.copytree(lit_repo, sub_dir, dirs_exist_ok=True)

        html_out = os.path.join(tmpdir, "main.html")
        short_hash = commit_hash[:7]

        # Get tag if any
        tag_proc = subprocess.run(
            ["git", "-C", repo_dir, "describe", "--tags", "--exact-match", commit_hash],
            capture_output=True,
            text=True,
        )
        git_tag = tag_proc.stdout.strip() if tag_proc.returncode == 0 else ""

        typst_cmd = [
            "typst",
            "compile",
            "--ignore-system-fonts",
            "--font-path",
            os.path.join(tmpdir, "src"),
            "--features",
            "html",
            "--format",
            "html",
            "--input",
            f"GIT_HASH={short_hash}",
            "--input",
            f"GIT_TAG={git_tag}",
            "--root",
            tmpdir,
            src_main,
            html_out,
        ]

        t_res = subprocess.run(typst_cmd, capture_output=True, text=True)
        if t_res.returncode != 0:
            return None

        pandoc_cmd = [
            "pandoc",
            html_out,
            "-t",
            "plain",
            "--wrap",
            "none",
        ]
        p_res = subprocess.run(pandoc_cmd, capture_output=True, text=True)
        if p_res.returncode != 0:
            return None

        # Count words
        words = p_res.stdout.split()
        return len(words)


def generate_svg_plot(entries, output_svg_path, title="Thesis Word Count Over Time"):
    """
    Renders an SVG chart visualizing the word count history without external libraries.
    """
    if not entries:
        return

    valid = [e for e in entries if e.get("wc") is not None]
    if not valid:
        return

    width = 1000
    height = 550
    margin_top = 70
    margin_bottom = 90
    margin_left = 90
    margin_right = 50

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    times = [e["timestamp_unix"] for e in valid]
    counts = [e["wc"] for e in valid]

    min_t = min(times)
    max_t = max(times)
    if min_t == max_t:
        max_t = min_t + 86400  # avoid division by zero

    min_c = 0
    max_c = max(counts)
    if max_c == 0:
        max_c = 1000
    # Add 10% headroom
    max_c = int(max_c * 1.1)

    def x_coord(t):
        return margin_left + (t - min_t) / (max_t - min_t) * plot_width

    def y_coord(c):
        return margin_top + plot_height - (c - min_c) / (max_c - min_c) * plot_height

    # Coordinates for points
    points = [(x_coord(e["timestamp_unix"]), y_coord(e["wc"])) for e in valid]

    # Polyline string
    polyline_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area_points = f"{points[0][0]:.1f},{margin_top + plot_height:.1f} " + polyline_str + f" {points[-1][0]:.1f},{margin_top + plot_height:.1f}"

    # Generate Y grid ticks (e.g. 5 steps)
    y_ticks = []
    tick_step = max_c // 5 or 1
    power = 10 ** (len(str(tick_step)) - 1)
    tick_step = max(1, (tick_step // power) * power)
    curr_tick = 0
    while curr_tick <= max_c:
        y_ticks.append(curr_tick)
        curr_tick += tick_step

    y_grid_elements = []
    for val in y_ticks:
        y = y_coord(val)
        y_grid_elements.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#2e3440" stroke-width="1" stroke-dasharray="4"/>'
            f'<text x="{margin_left - 15}" y="{y + 4:.1f}" fill="#8892b0" font-size="12" text-anchor="end" font-family="system-ui, sans-serif">{val:,}</text>'
        )

    # Generate X grid ticks (e.g. 5-7 dates)
    x_grid_elements = []
    num_x_ticks = 6
    dt_step = (max_t - min_t) / num_x_ticks
    for i in range(num_x_ticks + 1):
        t = min_t + i * dt_step
        x = x_coord(t)
        date_str = datetime.datetime.fromtimestamp(t).strftime("%d.%b %Y")
        x_grid_elements.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#2e3440" stroke-width="1" stroke-dasharray="4"/>'
            f'<text x="{x:.1f}" y="{margin_top + plot_height + 25}" fill="#8892b0" font-size="11" text-anchor="middle" font-family="system-ui, sans-serif">{date_str}</text>'
        )

    # Markers for points
    dots = []
    for i, e in enumerate(valid):
        x, y = points[i]
        date_str = datetime.datetime.fromtimestamp(e["timestamp_unix"]).strftime("%Y-%m-%d %H:%M")
        tooltip = f"{e['hash'][:7]} ({date_str}): {e['wc']} words"
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#38bdf8" stroke="#0f172a" stroke-width="2">'
            f'<title>{html.escape(tooltip)}</title>'
            f'</circle>'
        )

    last_entry = valid[-1]
    latest_wc = last_entry["wc"]
    latest_hash = last_entry["hash"][:7]

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
  <defs>
    <linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="#0b0f19" rx="12"/>

  <!-- Title & Stats -->
  <text x="{margin_left}" y="42" fill="#f8fafc" font-size="20" font-weight="600" font-family="system-ui, sans-serif">{html.escape(title)}</text>
  <text x="{width - margin_right}" y="42" fill="#38bdf8" font-size="16" font-weight="500" text-anchor="end" font-family="system-ui, sans-serif">Latest: {latest_wc:,} words ({latest_hash})</text>

  <!-- Grid lines -->
  {''.join(y_grid_elements)}
  {''.join(x_grid_elements)}

  <!-- Axes -->
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#475569" stroke-width="2"/>
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#475569" stroke-width="2"/>

  <!-- Area under curve -->
  <polygon points="{area_points}" fill="url(#areaGrad)"/>

  <!-- Plot line -->
  <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{polyline_str}"/>

  <!-- Point dots -->
  {''.join(dots)}

  <!-- Axis labels -->
  <text x="{margin_left - 60}" y="{margin_top + plot_height / 2}" fill="#94a3b8" font-size="13" font-weight="500" text-anchor="middle" font-family="system-ui, sans-serif" transform="rotate(-90 {margin_left - 60} {margin_top + plot_height / 2})">Word Count</text>
  <text x="{margin_left + plot_width / 2}" y="{margin_top + plot_height + 60}" fill="#94a3b8" font-size="13" font-weight="500" text-anchor="middle" font-family="system-ui, sans-serif">Commit Date</text>
</svg>
"""
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)


def default_repo_dir():
    # If run inside thesis repository, use current dir.
    # If run in thesis-practice/src/905_..., check sibling repo thesis.
    candidates = [
        os.getcwd(),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../thesis")),
        os.path.expanduser("~/Documents/repos/thesis"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, ".git")):
            return c
    return os.getcwd()


def main():
    parser = argparse.ArgumentParser(
        description="Compute word count per commit via Typst/Pandoc and export JSON + SVG plot."
    )
    parser.add_argument(
        "--repo-dir",
        "-r",
        default=default_repo_dir(),
        help="Path to thesis git repository (default: detected thesis repo path)",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default="wordcount_history.json",
        help="Path for exported JSON file (default: wordcount_history.json)",
    )
    parser.add_argument(
        "--output-plot",
        "-p",
        default="wordcount_plot.svg",
        help="Path for exported plot (default: wordcount_plot.svg)",
    )
    parser.add_argument(
        "--cache",
        "-c",
        default=".wordcount_cache.json",
        help="Path to cache file to store per-commit counts (default: .wordcount_cache.json)",
    )
    parser.add_argument(
        "--rev-range",
        default="HEAD",
        help="Git revision or range (e.g., HEAD, master, origin/main, HEAD~30..HEAD)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass existing cache",
    )
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        print(f"Error: Not a git repository: {repo_dir}", file=sys.stderr)
        sys.exit(1)

    # Load cache if available
    cache = {}
    if not args.no_cache and os.path.exists(args.cache):
        try:
            with open(args.cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}", file=sys.stderr)

    commits = get_commit_list(repo_dir, args.rev_range, reverse=True)
    print(f"Repository: {repo_dir}")
    print(f"Found {len(commits)} commits to inspect.")

    to_compute = [c for c in commits if c["hash"] not in cache]
    print(f"{len(to_compute)} commits to compute ({len(commits) - len(to_compute)} cached).")

    if to_compute:
        print(f"Processing using {args.workers} worker(s)...")

        def worker_task(c):
            h = c["hash"]
            wc = compute_commit_wc(repo_dir, h)
            return h, wc

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_commit = {
                executor.submit(worker_task, c): c for c in to_compute
            }
            for future in concurrent.futures.as_completed(future_to_commit):
                c = future_to_commit[future]
                try:
                    h, wc = future.result()
                    cache[h] = wc
                except Exception as ex:
                    print(f"Error processing {c['hash'][:7]}: {ex}", file=sys.stderr)
                    cache[c["hash"]] = None

                completed += 1
                if completed % 10 == 0 or completed == len(to_compute):
                    print(f"Progress: {completed}/{len(to_compute)} commits processed...")

        try:
            with open(args.cache, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}", file=sys.stderr)

    result_array = []
    for c in commits:
        wc = cache.get(c["hash"])
        result_array.append({
            "hash": c["hash"],
            "timestamp": c["timestamp"],
            "timestamp_unix": c["timestamp_unix"],
            "wc": wc,
        })

    export_json = [
        {
            "hash": item["hash"],
            "timestamp": item["timestamp"],
            "wc": item["wc"],
        }
        for item in result_array
    ]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(export_json, f, indent=2)
    print(f"Saved JSON export to {args.output_json} ({len(export_json)} entries)")

    generate_svg_plot(result_array, args.output_plot)
    print(f"Saved SVG plot to {args.output_plot}")


if __name__ == "__main__":
    main()
