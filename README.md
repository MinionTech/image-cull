# Image Cull

A local CLI tool for culling, deduplicating, and sorting photo collections and AI generations using local vision LLMs via [Ollama](https://ollama.com).

Evaluates image quality, detects generation artifacts and defects (e.g. plastic skin, warped fingers, lighting errors), scores quality from 1.0 to 10.0, and automatically filters unwanted photos into a target directory while preserving keepers in-place.

---

## Features

- **Local & Private:** Queries local vision models (`llava`, `llama3.2-vision`, `qwen2.5-vl`) running via Ollama — zero cloud API costs or data leakage.
- **Structured Pydantic Output:** Enforces JSON schema validation for reliable numerical scores, boolean flags, artifact tags, and forensic reasoning.
- **Automated Image Sorting:** Preserves quality images in-place and moves filtered files to `--filter-dir`.
- **Dry-Run Mode:** Generate diagnostic JSON reports without moving any image files.
- **Containerized CLI:** Pre-packaged wrapper for Podman / Docker to run like a native binary from anywhere on your system.
- **HEIC / HEIF support:** iPhone and Google Takeout photos (`.heic`, `.heif`) via `pillow-heif`; the container image includes `libheif`.

---

## Quickstart

### 1. Prerequisites
- Podman or Docker installed. (No bare-metal Ollama daemon installation required; `image-cull` automatically runs and manages a containerized Ollama backend).

Supported input formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.heic`, `.heif`. HEIC decoding uses `pillow-heif` (registered at startup). The Docker image installs `libheif1` for HEIF decode in the container; local runs need `pillow-heif` from `requirements.txt` (and on Linux, system `libheif` if wheels are unavailable).

### 2. Installation
Clone the repository and run the setup script:

```bash
git clone https://github.com/DerekRoberts/image-cull.git
cd image-cull
./setup.sh
```

The `./setup.sh` script builds the `image-cull` container image, pre-caches the Ollama container, and installs the standalone `image-cull` binary wrapper into `~/.local/bin/image-cull`.

### 3. Backend Lifecycle
- **Zero Configuration:** When you run `image-cull`, the wrapper checks if an Ollama service is reachable. If not, it automatically spawns a background `image-cull-ollama` container, runs your command, and stops the backend on exit.
- **Model Persistence:** Downloaded vision models (`llava`, `llama3.2-vision`, `qwen2.5-vl`) persist across runs inside a named volume (`image-cull-ollama-models`).
- **Remote / Pre-existing Ollama:** If you already run Ollama on host or a remote server, export `OLLAMA_HOST="http://<ip>:<port>"` and `image-cull` will use that endpoint instead.

---


## Usage

### Run from anywhere
```bash
image-cull ~/Downloads --threshold 7.5
```

### Dry-run (JSON report only, no file movements)
```bash
image-cull ~/Downloads --dry-run
```

### Apply moves from a reviewed report (no Ollama)
After reviewing or hand-editing `cull-report.json` (legacy names `cull_report.json`, `realism_audit_report.json` also work):
```bash
image-cull ~/Downloads --apply-report
image-cull ~/Downloads --apply-report --threshold 7.5
image-cull ~/Downloads --apply-report --threshold-quality 6.0 --threshold-ai 7.5
image-cull ~/Downloads --apply-report --force-reapply
```

Apply reads cutoffs from `meta.thresholds`; CLI flags override. Rejects when **any populated dimension** with an active threshold fails. Logs the reason per file.

Successful moves set `applied: true` and `applied_at` (ISO UTC) on each result entry and persist the report back to the same path. Re-running `--apply-report` skips entries already marked applied; use `--force-reapply` to process them again (if the source file is already gone, apply logs a warning and skips that entry).

### Specify custom vision model
```bash
image-cull ~/Downloads --model llava --threshold 8.0
```

### Cull profiles (`--profile`)

Profiles select which analysis lenses run and set default thresholds. Without `--profile`, behavior is unchanged: single AI realism score, legacy `analysis` block in the report, no `meta.profile`.

| Profile | Question | Lenses | Status |
| --- | --- | --- | --- |
| `mixed` (recommended for unknown folders) | What is this and should I keep it? | `hygiene`, `ai` (+ `quality` when #19 lands) | **Hygiene + AI lenses working** |
| `ai-fun` | Is this render successful / worth keeping? | `generation` (+ optional `hygiene`) | **Generation lens working** |
| `photos` | Is this a keeper real photo? | `hygiene`, `quality` | **Hygiene + quality lenses working** |

**Mixed folder (fully working today):**
```bash
image-cull ~/Downloads --profile mixed --dry-run
image-cull ~/Downloads --profile mixed --threshold 7.5
```

Reports use the multi-dimensional layout: `meta.profile`, `meta.thresholds`, and per-file `ai` blocks (not legacy `analysis`).

**AI art folder (`ai-fun` — generation success, not photorealism):**

For intentional AI art, photorealism is the wrong metric. The generation lens scores whether a render *succeeded* and is worth keeping — not whether it looks like a photograph.

| Image | Realism score (wrong lens) | Generation success (right lens) |
| --- | --- | --- |
| Octo-rex mashup, coherent | Low | **High** — creative, readable subjects |
| T-rex with six fingers | Low | **Low** — anatomical failure |
| Melted face / garbled text | Low | **Low** — broken render |
| Stylized but clean cartoon | Low | **High** — if that's the intent |

```bash
image-cull ~/AI-Art --profile ai-fun --dry-run
image-cull ~/AI-Art --profile ai-fun --threshold 7.5
image-cull ~/AI-Art --profile ai-fun --fast --max-dimension 1024 --dry-run
```

Reports use `generation` blocks with `success_score`, `issues[]`, and `reasoning`. Rejects when `success_score` is below `--threshold-generation` (or `--threshold`, which maps to the profile's primary lens). Set `"keep": true` on edge cases after dry-run review to preserve them on `--apply-report`.

**Camera roll / Takeout (keeper quality, not AI detection):**

For real photos, the question is *"Would I keep this in an album?"* — not whether it looks AI-generated. The quality lens scores blur, exposure, framing accidents, and screenshots.

| Image | AI realism (wrong lens) | Keeper quality (right lens) |
| --- | --- | --- |
| Sharp sunset, well exposed | High | **High** — intentional keeper |
| Motion-blurred party shot | High | **Low** — motion_blur |
| Pocket / floor accidental capture | High | **Low** — pocket_shot, accidental_frame |
| Screenshot with UI chrome | High | **Low** — screenshot, ui_chrome |
| Eyes closed on group photo | High | **Low** — eyes_closed |

```bash
image-cull ~/Pictures --profile photos --dry-run
image-cull ~/Pictures --profile photos --threshold 6.0
image-cull ~/Pictures --profile photos --fast --max-dimension 1024 --dry-run
```

Reports use `quality` blocks with `keeper_score`, `issues[]`, and `reasoning`. Rejects when `keeper_score` is below `--threshold-quality` (or `--threshold`, which maps to the profile's primary lens). Common issue tags: `motion_blur`, `out_of_focus`, `underexposed`, `overexposed`, `eyes_closed`, `accidental_frame`, `finger_on_lens`, `pocket_shot`, `floor_shot`, `screenshot`, `ui_chrome`, `duplicate_feel`. Set `"keep": true` on edge cases after dry-run review to preserve them on `--apply-report`.

Override the profile’s lens set (still validates implementation):
```bash
image-cull ~/Downloads --profile mixed --checks ai --dry-run
```

Per-dimension threshold overrides (defaults come from the profile):
```bash
image-cull ~/Downloads --profile mixed --threshold 7.5 --threshold-quality 6.0 --dry-run
```

**Hygiene pre-filter (dupes, corruption, min-res):**

Runs before Ollama on `mixed` and `photos` profiles (or via `--checks hygiene`). Exact duplicates are detected by SHA-256; the first file by sorted name is kept, later copies get `exact_dupe_of`. Corrupt headers, solid-color blanks, and undersized images reject without VLM calls.

```bash
image-cull ~/Downloads --profile mixed --min-res 512x512 --dry-run
image-cull ~/AI-Art --profile ai-fun --checks hygiene,generation --min-res 512x512 --dry-run
```

---

## CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `directory` | `.` | Target input directory containing images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.heic`, `.heif`) |
| `--filter-dir` | `<input_dir>/rejects` | Directory to move filtered-out/rejected files into |
| `--model` | `llava` | Local vision model to query via Ollama |
| `--threshold` | `7.0` (dry-run only) | Minimum score for the profile’s primary lens, or AI realism when no `--profile`; with `--profile`, profile default applies if omitted |
| `--threshold-ai` | — | Override AI realism cutoff (1.0 to 10.0) |
| `--threshold-quality` | — | Override quality keeper cutoff (1.0 to 10.0) |
| `--threshold-generation` | — | Override generation success cutoff (1.0 to 10.0) |
| `--profile` | — | `mixed`, `ai-fun`, or `photos`; omit for legacy single-score mode |
| `--checks` | — | Comma-separated lenses overriding the profile set (`hygiene`, `ai`, `quality`, `generation`) |
| `--min-res` | — | Reject images below WIDTHxHEIGHT pixels (e.g. `512x512`); hygiene lens only |
| `--dry-run` | `False` | Generate report without moving files |
| `--apply-report` | — | Apply file moves from an existing cull report without re-analyzing (optional path; default: first of `cull-report.json`, `cull_report.json`, `realism_audit_report.json` in input dir) |
| `--force-reapply` | `False` | With `--apply-report`, re-process entries already marked `applied: true` (default: skip them) |
| `--max-dimension` | `0` (off) | Optional: downscale before Ollama so the long edge is at most N px (`0` = send originals; **default**) |
| `--fast` | `False` | Minimal VLM output (score, flag, artifacts only); skips reasoning for faster bulk triage |

### Fast vs full mode (`--fast`)

**Full mode (default)** asks the model for forensic reasoning alongside score, realism flag, and artifact tags. Use this for dry-runs, borderline decisions, and any run where you will read the JSON report before moving files.

**Fast mode** (`--fast`) uses a reduced prompt and schema — score and artifact tags only, no model reasoning. The report still includes a `reasoning` key (empty string) so the format stays consistent; `meta.fast` is `true`. Works for `ai` (realism), `generation` (success), and `quality` (keeper) lenses. Speed gains depend on the model, image size, and hardware.

**When to use `--fast`:**

- Bulk triage on large folders where obvious rejects/keepers dominate
- A first pass before `--apply-report`, with manual `keep` overrides on edge cases
- Combined with `--dry-run` to generate scores quickly, then re-run borderline files at full resolution

**When to stay on full mode:**

- First cull of a new collection or model
- Scores near your threshold (±1.0) where reasoning helps you decide overrides
- Any workflow where the JSON report is the primary review surface

`--fast` composes with `--dry-run`, parallel Ollama pipelining, and `--max-dimension`.

### Speed vs accuracy (`--max-dimension`)

**Off by default.** Omit the flag (or pass `--max-dimension 0`) to send full-resolution originals — same behavior as before this option existed.

Opt in only when you've measured a speed win on large files and accept the accuracy tradeoffs below. Vision models typically downsample internally (~336–1024 px on the long edge), so `--max-dimension 1024` may shave ~10–30% off 4K+ images with little change for obvious rejects/keepers.

**When not to use this:**

- First pass on a new collection — start at full res until you know where scores land.
- Borderline keepers (roughly threshold ± 1.0) — micro-artifacts (teeth, fingers, hair, small text) may disappear and inflate scores.
- Already-small or upscaled AI images — further downscaling removes the detail that reveals fakeness.
- When JPEG re-encode noise could matter — downscaling re-encodes JPEGs and can add blockiness that looks “AI.”

If you do enable it, dry-run both ways on a sample folder and compare scores before trusting moves. Re-check any borderline keepers at full resolution.

---

## Sample JSON Cull Report (`cull-report.json`)

### Single-score mode (default)

Current runs emit one `analysis` block per file plus legacy `meta.threshold`. New reports also record `meta.thresholds` for forward compatibility with profiles.

```json
{
  "meta": {
    "threshold": 7.0,
    "thresholds": { "ai": 7.0, "quality": null, "generation": null },
    "model": "llava",
    "max_dimension": 0,
    "fast": false,
    "generated_at": "2026-07-29T12:00:00+00:00",
    "dry_run": true
  },
  "results": [
    {
      "file": "beach_portrait.jpg",
      "analysis": {
        "realism_score": 8.5,
        "is_realistic": true,
        "detected_artifacts": [
          "AI-generated background texture",
          "Overly uniform skin smoothness"
        ],
        "reasoning": "The image features realistic lighting and natural human posture. However, minor AI artifacts are present in the hair texture and background foliage."
      }
    },
    {
      "file": "broken.jpg",
      "status": "error",
      "error": "Error processing broken.jpg: ...",
      "keep": null
    }
  ]
}
```

Each result may include an optional `keep` field (`true` = never move, `false` = always move). Failed entries require `keep` before apply will move them. After a successful move, apply sets `applied: true` and `applied_at` (ISO UTC) on that entry. Legacy bare-array reports are still accepted on read.

### Multi-dimensional layout (`--profile`)

With `--profile`, each result carries **only the populated dimension blocks** — no empty placeholders. `meta.profile` names the active profile; `meta.thresholds` holds per-dimension cutoffs (`null` = lens inactive for this run).

| Block | Purpose | Key fields |
| --- | --- | --- |
| `hygiene` | Deterministic pre-filter (#3): exact dupes (SHA-256), corruption, min-res, solid-color blanks | `action` (`reject` \| `keep`), optional `exact_dupe_of`, optional `reason` |
| `ai` | Photorealism / AI-artifact detection | `realism_score`, `issues[]`, `reasoning` |
| `quality` | Real-photo keeper scoring (blur, exposure, framing) | `keeper_score`, `issues[]`, `reasoning` |
| `generation` | AI-art success (subject landed, not melted) | `success_score`, `issues[]`, `reasoning` |

```json
{
  "meta": {
    "profile": "mixed",
    "threshold": 7.0,
    "thresholds": { "ai": 7.0, "quality": 6.0, "generation": null },
    "model": "llava",
    "max_dimension": 0,
    "fast": false,
    "generated_at": "2026-07-29T12:00:00+00:00",
    "dry_run": true
  },
  "results": [
    {
      "file": "beach_portrait.jpg",
      "ai": {
        "realism_score": 8.5,
        "is_realistic": true,
        "issues": [
          "AI-generated background texture",
          "Overly uniform skin smoothness"
        ],
        "reasoning": "The image features realistic lighting and natural human posture."
      }
    }
  ]
}
```

Example with multiple lenses (`mixed` / `ai-fun` / `photos`):

```json
{
  "meta": {
    "profile": "ai-fun",
    "thresholds": { "ai": null, "quality": 6.0, "generation": 7.0 },
    "generated_at": "2026-07-29T12:00:00+00:00",
    "dry_run": true
  },
  "results": [
    {
      "file": "octo-rex.png",
      "generation": {
        "success_score": 8.5,
        "issues": [],
        "reasoning": "Creative mashup, coherent subjects, no garbled anatomy."
      },
      "keep": null
    },
    {
      "file": "IMG_blur.jpg",
      "quality": {
        "keeper_score": 3.2,
        "issues": ["motion_blur", "underexposed"],
        "reasoning": "Subject motion blur and heavy underexposure."
      },
      "keep": null
    },
    {
      "file": "copy.jpg",
      "hygiene": {
        "exact_dupe_of": "original.jpg",
        "action": "reject"
      }
    }
  ]
}
```

**Backward compatibility:** reports with top-level `analysis` / `realism_score` still load and apply using `meta.threshold` (or `meta.thresholds.ai`). Multi-dimensional blocks use per-dimension thresholds from `meta.thresholds` (CLI overrides win). Composite apply rejects when **any populated dimension** with an active threshold fails; honors `hygiene.action` and `keep` overrides. Apply logs the reject/preserve reason per file.

**Applied state:** `--apply-report` writes the report back after each run. Moved entries gain `applied: true` and `applied_at`. `meta.last_applied_at` and `meta.last_applied_thresholds` record when apply last ran and which cutoffs were used. Re-apply skips `applied: true` entries unless `--force-reapply` is set.

---

## Self-check & CI

**Before push:** `./check.sh`

That script is the local entrypoint; CI runs `./check.sh host` and `./check.sh docker` in parallel (plus an advisory Trivy scan that stays GitHub-only for SARIF upload). Host needs:

- Python 3 on PATH as `python` or `python3`. App deps belong in a venv (PEP 668 on modern distros): `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Linux: system `libheif` (e.g. `libheif1` / `libheif`) for HEIF decode
- `ruff` at the pin in `check.sh` (if missing/mismatched, the script installs it into `.venv/` — never into the system interpreter)
- `shellcheck`
- Working Docker, or Podman as fallback

```bash
./check.sh          # host + docker
./check.sh host     # self-check, ruff, shellcheck
./check.sh docker   # image build + container --self-check
```

Individual pieces if you only need one:

```bash
python image_cull.py --self-check
ruff check image_cull.py
shellcheck setup.sh check.sh
docker build -t image-cull:local . && docker run --rm image-cull:local --self-check
```

`--self-check` includes committed hygiene fixtures under `fixtures/hygiene/` (see `manifest.json` for expected `check_hygiene()` outcomes and a hygiene-only dry-run). No Ollama required.

The `Results` job in `.github/workflows/ci.yml` is the intended single required status check on `main`.

---

## License

MIT
