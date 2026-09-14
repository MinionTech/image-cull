from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import ollama
from pydantic import BaseModel, Field, field_validator

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
HEIF_SUFFIXES = frozenset({".heic", ".heif"})


def _register_heif_opener() -> None:
    from pillow_heif import register_heif_opener

    register_heif_opener()


_register_heif_opener()
DEFAULT_THRESHOLD = 7.0
DEFAULT_REPORT_NAME = "cull-report.json"
LEGACY_REPORT_NAMES = ("cull-report.json", "cull_report.json", "realism_audit_report.json")
MAX_IN_FLIGHT = 16  # ponytail: cap client-side requests; Ollama throttles GPU work
SCORE_FIELD = {"ai": "realism_score", "quality": "keeper_score", "generation": "success_score"}
DIMENSION_BLOCKS = ("hygiene", "ai", "quality", "generation")
SCORED_DIMENSIONS = ("ai", "quality", "generation")
PROFILE_NAMES = ("mixed", "ai-fun", "photos")
IMPLEMENTED_LENSES = frozenset({"ai", "generation", "quality", "hygiene"})
LENS_ISSUE = {"generation": "#18", "quality": "#19", "hygiene": "#3"}
_ISSUE_TAG = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


class CullConfig:
    """Resolved profile, active lenses, and per-dimension thresholds."""

    __slots__ = ("lenses", "multi_dimensional", "profile", "thresholds")

    def __init__(
        self,
        *,
        profile: str | None,
        lenses: tuple[str, ...],
        thresholds: dict[str, float | None],
    ):
        self.profile = profile
        self.lenses = lenses
        self.thresholds = thresholds
        self.multi_dimensional = profile is not None


PROFILE_DEFAULTS: dict[str, dict] = {
    "mixed": {
        "lenses": ("hygiene", "ai"),
        "thresholds": {"ai": 7.0, "quality": 6.0, "generation": None},
    },
    "ai-fun": {
        "lenses": ("generation",),
        "thresholds": {"ai": None, "quality": None, "generation": 7.0},
    },
    "photos": {
        "lenses": ("hygiene", "quality"),
        "thresholds": {"ai": None, "quality": 6.0, "generation": None},
    },
}

PROFILE_PRIMARY_THRESHOLD = {"mixed": "ai", "ai-fun": "generation", "photos": "quality"}


class ScanProgress:
    """Thread-safe [n/total] prefix for parallel full scans."""

    def __init__(self, total: int):
        self.total = total
        self._lock = threading.Lock()
        self._next = 0

    def begin(self) -> str:
        with self._lock:
            self._next += 1
            return f"[{self._next}/{self.total}]"


class RealismAnalysis(BaseModel):
    realism_score: float
    is_realistic: bool
    detected_artifacts: list[str]
    reasoning: str


class RealismAnalysisFast(BaseModel):
    realism_score: float
    is_realistic: bool
    detected_artifacts: list[str]


class HygieneVerdict(BaseModel):
    action: str
    exact_dupe_of: str | None = None
    reason: str | None = None


class AiVerdict(BaseModel):
    realism_score: float
    is_realistic: bool | None = None
    issues: list[str] = []
    reasoning: str = ""


class _QualityVerdictCore(BaseModel):
    keeper_score: float = Field(ge=1.0, le=10.0)
    issues: list[str] = []

    @field_validator("keeper_score")
    @classmethod
    def keeper_score_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("keeper_score must be finite")
        return v

    @field_validator("issues")
    @classmethod
    def issues_snake_case(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if not _ISSUE_TAG.match(t)]
        if bad:
            raise ValueError(f"issues must be snake_case tags, got: {bad}")
        return v


class QualityVerdict(_QualityVerdictCore):
    reasoning: str = ""


class QualityVerdictFast(_QualityVerdictCore):
    pass


class GenerationVerdict(BaseModel):
    success_score: float
    issues: list[str] = []
    reasoning: str = ""


class GenerationVerdictFast(BaseModel):
    success_score: float
    issues: list[str] = []


def default_thresholds(threshold: float) -> dict[str, float | None]:
    return {"ai": threshold, "quality": None, "generation": None}


def empty_thresholds() -> dict[str, float | None]:
    return {"ai": None, "quality": None, "generation": None}


def parse_min_res(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    match = re.match(r"^(\d+)x(\d+)$", raw.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("--min-res must be WIDTHxHEIGHT (e.g. 512x512)")
    width, height = int(match.group(1)), int(match.group(2))
    if width < 1 or height < 1:
        raise ValueError("--min-res dimensions must be positive")
    return width, height


def parse_checks(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    lenses = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not lenses:
        raise ValueError("--checks requires at least one lens name")
    unknown = [l for l in lenses if l not in DIMENSION_BLOCKS]
    if unknown:
        raise ValueError(f"unknown lens(es): {', '.join(unknown)}")
    return lenses


def resolve_cull_config(
    *,
    profile: str | None,
    checks: tuple[str, ...] | None,
    threshold: float,
    threshold_ai: float | None,
    threshold_quality: float | None,
    threshold_generation: float | None,
) -> CullConfig:
    if profile is None:
        thresholds = default_thresholds(threshold)
        if threshold_ai is not None:
            thresholds["ai"] = threshold_ai
        if threshold_quality is not None:
            thresholds["quality"] = threshold_quality
        if threshold_generation is not None:
            thresholds["generation"] = threshold_generation
        lenses = checks if checks is not None else ("hygiene", "ai")
        return CullConfig(profile=None, lenses=lenses, thresholds=thresholds)

    defaults = PROFILE_DEFAULTS[profile]
    lenses = checks if checks is not None else defaults["lenses"]
    thresholds = dict(empty_thresholds())

    if checks is None:
        for dim, value in defaults["thresholds"].items():
            if value is not None:
                thresholds[dim] = value
        primary = PROFILE_PRIMARY_THRESHOLD[profile]
        if primary in lenses:
            thresholds[primary] = threshold
    else:
        scored_active = [l for l in lenses if l in SCORED_DIMENSIONS]
        if scored_active:
            thresholds[scored_active[0]] = threshold

    if threshold_ai is not None:
        thresholds["ai"] = threshold_ai
    if threshold_quality is not None:
        thresholds["quality"] = threshold_quality
    if threshold_generation is not None:
        thresholds["generation"] = threshold_generation

    return CullConfig(profile=profile, lenses=lenses, thresholds=thresholds)


def validate_cull_config(config: CullConfig) -> None:
    missing = [l for l in config.lenses if l not in IMPLEMENTED_LENSES]
    if not missing:
        return
    parts = []
    for lens in missing:
        issue = LENS_ISSUE.get(lens, "open issue")
        parts.append(f"  - {lens}: not implemented yet (see {issue})")
    profile_note = f" (profile={config.profile})" if config.profile else ""
    raise SystemExit(
        "Cannot run cull: requested lens(es) are not implemented"
        f"{profile_note}:\n"
        + "\n".join(parts)
        + "\nUse --profile mixed for AI realism scoring, or --checks to override lenses."
    )


def effective_thresholds(meta: dict | None, cli_threshold: float) -> dict[str, float | None]:
    if meta and "thresholds" in meta:
        thresholds = dict(meta["thresholds"])
        if thresholds.get("ai") is None:
            legacy = meta.get("threshold")
            if legacy is not None:
                thresholds["ai"] = legacy
        return thresholds
    legacy = (meta or {}).get("threshold", cli_threshold)
    return default_thresholds(legacy)


def resolve_apply_thresholds(
    meta: dict | None,
    *,
    threshold: float | None,
    threshold_ai: float | None,
    threshold_quality: float | None,
    threshold_generation: float | None,
) -> dict[str, float | None]:
    """Merge meta.thresholds with CLI flags; per-dimension overrides win."""
    fallback = threshold if threshold is not None else DEFAULT_THRESHOLD
    thresholds = effective_thresholds(meta, fallback)

    if threshold is not None:
        active = [d for d in SCORED_DIMENSIONS if thresholds.get(d) is not None]
        if len(active) == 1:
            thresholds[active[0]] = threshold
        else:
            profile = (meta or {}).get("profile")
            if profile in PROFILE_PRIMARY_THRESHOLD:
                thresholds[PROFILE_PRIMARY_THRESHOLD[profile]] = threshold
            else:
                thresholds["ai"] = threshold

    if threshold_ai is not None:
        thresholds["ai"] = threshold_ai
    if threshold_quality is not None:
        thresholds["quality"] = threshold_quality
    if threshold_generation is not None:
        thresholds["generation"] = threshold_generation
    return thresholds


def resolve_report_path(input_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for name in LEGACY_REPORT_NAMES:
        candidate = input_dir / name
        if candidate.is_file():
            return candidate
    return input_dir / DEFAULT_REPORT_NAME


def multi_dimensional_active(thresholds: dict[str, float | None]) -> bool:
    return sum(v is not None for v in thresholds.values()) > 1


def analysis_to_ai_block(analysis: dict) -> dict:
    return {
        "realism_score": analysis["realism_score"],
        "is_realistic": analysis.get("is_realistic"),
        "issues": analysis.get("detected_artifacts", []),
        "reasoning": analysis.get("reasoning", ""),
    }


def dimension_score(entry: dict, dimension: str) -> float | None:
    block = entry.get(dimension)
    if not block:
        return None
    if dimension == "ai":
        return block.get("realism_score")
    if dimension == "quality":
        return block.get("keeper_score")
    if dimension == "generation":
        return block.get("success_score")
    return None


def legacy_analysis_score(entry: dict) -> float | None:
    analysis = entry.get("analysis")
    if analysis and "realism_score" in analysis:
        return analysis["realism_score"]
    return dimension_score(entry, "ai")


def build_report_meta(
    *,
    threshold: float,
    model: str,
    max_dimension: int,
    fast: bool,
    dry_run: bool,
    profile: str | None = None,
    thresholds: dict[str, float | None] | None = None,
    min_res: tuple[int, int] | None = None,
) -> dict:
    meta = {
        "threshold": threshold,
        "thresholds": thresholds or default_thresholds(threshold),
        "model": model,
        "max_dimension": max_dimension,
        "fast": fast,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
    }
    if profile is not None:
        meta["profile"] = profile
    if min_res is not None:
        meta["min_res"] = f"{min_res[0]}x{min_res[1]}"
    return meta


def result_entry_from_lenses(
    filename: str,
    *,
    hygiene: dict | None = None,
    ai: dict | None = None,
    quality: dict | None = None,
    generation: dict | None = None,
    keep: bool | None = None,
) -> dict:
    entry: dict = {"file": filename}
    for key, block in (
        ("hygiene", hygiene),
        ("ai", ai),
        ("quality", quality),
        ("generation", generation),
    ):
        if block is not None:
            entry[key] = block
    if keep is not None:
        entry["keep"] = keep
    return entry


def realism_result_entry(filename: str, analysis: dict, *, multi_dimensional: bool = False) -> dict:
    if multi_dimensional:
        return result_entry_from_lenses(filename, ai=analysis_to_ai_block(analysis))
    return {"file": filename, "analysis": analysis}


def parse_args():
    parser = argparse.ArgumentParser(description="Cull and sort AI-generated images based on photorealism using Ollama.")
    parser.add_argument("--dir", default="./photos", help="Input directory containing images (.png, .jpg, .jpeg, .webp)")
    parser.add_argument("--recursive", action="store_true", help="Recursively traverse input directory")
    parser.add_argument("--filter-dir", default=None, help="Directory to move filtered-out/rejected files into (default: <input_dir>/rejects)")
    parser.add_argument("--model", default="llava", help="Local vision model to query via Ollama")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum score (1.0 to 10.0) for the profile's primary lens, or AI realism when no --profile; "
        "required unless --dry-run or --profile supplies a default",
    )
    parser.add_argument(
        "--threshold-ai",
        type=float,
        default=None,
        metavar="N",
        help="Override AI realism cutoff (1.0 to 10.0)",
    )
    parser.add_argument(
        "--threshold-quality",
        type=float,
        default=None,
        metavar="N",
        help="Override quality keeper cutoff (1.0 to 10.0)",
    )
    parser.add_argument(
        "--threshold-generation",
        type=float,
        default=None,
        metavar="N",
        help="Override generation success cutoff (1.0 to 10.0)",
    )
    parser.add_argument(
        "--edit-policy",
        choices=["original", "edited", "both"],
        default=None,
        help="How to handle original vs edited file pairs (e.g. IMG_1234.jpg and IMG_1234-edited.jpg) (default: original)",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=None,
        help="Cull profile: mixed (AI realism), ai-fun (generation), photos (quality); "
        "default: legacy single-score AI mode (no meta.profile)",
    )
    parser.add_argument(
        "--checks",
        default=None,
        metavar="LENSES",
        help="Comma-separated lenses overriding the profile set (hygiene, ai, quality, generation)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate the JSON report without physically moving files into filter_dir")
    parser.add_argument(
        "--apply-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help="Apply file moves from a cull report without re-analyzing (default: <input_dir>/cull-report.json)",
    )
    parser.add_argument(
        "--force-reapply",
        action="store_true",
        help="Re-apply entries already marked applied (default: skip them)",
    )
    parser.add_argument("--report-path-display", default=None, help="Custom path string to display in the final report message")
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=0,
        metavar="N",
        help="Optional downscale before Ollama: long edge at most N px (0 = disabled, default)",
    )
    parser.add_argument(
        "--min-res",
        default=None,
        metavar="WxH",
        help="Reject images smaller than WIDTHxHEIGHT pixels (e.g. 512x512); hygiene lens only",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Minimal VLM output (score, flag, artifacts only) for faster bulk triage",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run built-in assertions (including HEIF) and exit; used by CI",
    )
    args = parser.parse_args()

    if args.self_check:
        return args

    if args.apply_report is not None and args.dry_run:
        parser.error("--apply-report cannot be used with --dry-run")

    try:
        args.checks = parse_checks(args.checks)
    except ValueError as e:
        parser.error(str(e))

    try:
        args.min_res = parse_min_res(args.min_res)
    except ValueError as e:
        parser.error(str(e))

    if args.profile is None and args.checks is not None:
        parser.error("--checks requires --profile")

    if args.dry_run:
        if args.threshold is None:
            if args.profile is not None:
                primary = PROFILE_PRIMARY_THRESHOLD[args.profile]
                args.threshold = PROFILE_DEFAULTS[args.profile]["thresholds"][primary]
            else:
                args.threshold = DEFAULT_THRESHOLD
    elif args.threshold is None:
        if args.apply_report is not None:
            pass  # ponytail: apply reads cutoffs from report meta; CLI overrides optional
        elif args.profile is not None:
            primary = PROFILE_PRIMARY_THRESHOLD[args.profile]
            args.threshold = PROFILE_DEFAULTS[args.profile]["thresholds"][primary]
        else:
            parser.error("--threshold is required when not using --dry-run or --apply-report")

    for name, value in (
        ("--threshold", args.threshold),
        ("--threshold-ai", args.threshold_ai),
        ("--threshold-quality", args.threshold_quality),
        ("--threshold-generation", args.threshold_generation),
    ):
        if value is not None and (not (1.0 <= value <= 10.0) or math.isnan(value)):
            parser.error(f"{name} must be between 1.0 and 10.0, got {value}")
    if args.max_dimension < 0:
        parser.error(f"--max-dimension must be >= 0, got {args.max_dimension}")
    return args


def load_report(path: Path) -> tuple[dict | None, list]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return None, data
    if isinstance(data, dict) and "results" in data:
        return data.get("meta"), data["results"]
    raise SystemExit(f"Error: Unrecognized report format in '{path}'")


def write_report(path: Path, meta: dict, results: list):
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)


def score_sort_key(entry: dict) -> tuple[float, str]:
    scores = [s for dim in SCORED_DIMENSIONS if (s := dimension_score(entry, dim)) is not None]
    legacy = legacy_analysis_score(entry)
    if legacy is not None:
        scores.append(legacy)
    score = max(scores) if scores else -1.0
    return (-score, entry.get("file", ""))


def pipeline_depth(image_count: int) -> int:
    return max(1, min(image_count, MAX_IN_FLIGHT))


def _entry_ai_block(entry: dict) -> dict:
    block = entry.get("ai")
    if block:
        return block
    return entry.get("analysis") or {}


def _format_hygiene_reason(hygiene: dict) -> str:
    action = hygiene.get("action", "reject")
    if dupe := hygiene.get("exact_dupe_of"):
        return f"hygiene: {action} (exact_dupe_of {dupe})"
    if reason := hygiene.get("reason"):
        return f"hygiene: {action} ({reason})"
    return f"hygiene: {action}"


def _format_score_reason(entry: dict, dimension: str, score: float, cutoff: float, *, failed: bool) -> str:
    block = entry.get(dimension) if dimension != "ai" else _entry_ai_block(entry)
    issues = block.get("issues") or block.get("detected_artifacts") or []
    label = SCORE_FIELD[dimension]
    prefix = f"{issues[0]}, " if issues else ""
    if failed:
        return f"{dimension}: {prefix}{label} {score:.1f} < {cutoff:.1f}"
    return f"{dimension}: {label} {score:.1f}"


def evaluate_entry(entry: dict, thresholds: dict[str, float | None]) -> tuple[bool | None, list[str]]:
    """Return (reject?, reasons). None = skip (error without keep override)."""
    keep = entry.get("keep")
    if keep is True:
        return False, ["keep: true override"]
    if keep is False:
        return True, ["keep: false override"]
    if entry.get("status") == "error":
        return None, []

    hygiene = entry.get("hygiene")
    hygiene_keep_reason: str | None = None
    if hygiene:
        action = hygiene.get("action")
        reason = _format_hygiene_reason(hygiene)
        if action == "reject":
            return True, [reason]
        if action == "keep":
            hygiene_keep_reason = reason

    fail_reasons: list[str] = []
    pass_reasons: list[str] = []
    scored = False
    for dim in SCORED_DIMENSIONS:
        cutoff = thresholds.get(dim)
        if cutoff is None:
            continue
        score = dimension_score(entry, dim)
        if score is None and dim == "ai":
            score = _entry_ai_block(entry).get("realism_score")
        if score is None:
            continue
        scored = True
        if score < cutoff:
            fail_reasons.append(_format_score_reason(entry, dim, score, cutoff, failed=True))
        else:
            pass_reasons.append(_format_score_reason(entry, dim, score, cutoff, failed=False))

    if fail_reasons:
        return True, fail_reasons
    if scored:
        return False, pass_reasons
    if hygiene_keep_reason is not None:
        return False, [hygiene_keep_reason]
    return None, []


def should_reject(entry: dict, threshold: float, meta: dict | None = None) -> bool | None:
    thresholds = effective_thresholds(meta, threshold)
    reject, _ = evaluate_entry(entry, thresholds)
    return reject


def unique_reject_path(filter_dir: Path, filename: str) -> Path:
    dest = filter_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while (candidate := filter_dir / f"{stem}_{n}{suffix}").exists():
        n += 1
    return candidate


def unique_reject_dest_pair(dest_dir: Path, img_name: str, sidecar: Path) -> tuple[Path, Path]:
    """Pick image/sidecar destinations together so both keep a matching basename."""
    stem = Path(img_name).stem
    suffix = Path(img_name).suffix
    n = 1
    while n <= 10_000:
        candidate_name = img_name if n == 1 else f"{stem}_{n}{suffix}"
        img_dest = dest_dir / candidate_name
        sidecar_dest = dest_dir / _paired_sidecar_name(candidate_name, sidecar, img_name, n=n)
        if not img_dest.exists() and not sidecar_dest.exists():
            return img_dest, sidecar_dest
        n += 1
    raise RuntimeError(f"Could not allocate reject destination pair for {img_name}")


def _paired_sidecar_name(img_dest_name: str, sidecar: Path, src_img_name: str, *, n: int = 1) -> str:
    if sidecar.name.startswith(src_img_name):
        return f"{img_dest_name}{sidecar.name[len(src_img_name):]}"
    src_stem = Path(src_img_name).stem
    if sidecar.name.startswith(f"{src_stem}.") and sidecar.name.endswith(".json"):
        dest_stem = Path(img_dest_name).stem
        return f"{dest_stem}{sidecar.name[len(src_stem):]}"
    swapped = _bracket_swap_name(src_img_name)
    if swapped is not None and sidecar.name.startswith(swapped):
        dest_swapped = _bracket_swap_name(img_dest_name)
        if dest_swapped is not None:
            return f"{dest_swapped}{sidecar.name[len(swapped):]}"
    if n == 1:
        return sidecar.name
    if sidecar.name.endswith(".json"):
        return f"{sidecar.name[:-5]}_{n}.json"
    return f"{sidecar.name}_{n}"


def move_reject(img_path: Path, input_dir: Path, filter_dir: Path, move_lock: threading.Lock | None = None) -> Path:
    def _move() -> Path:
        resolved_img = img_path.resolve()
        resolved_input = input_dir.resolve()
        if not resolved_img.is_relative_to(resolved_input):
            raise ValueError(f"Path escapes input directory: {img_path}")
        rel = resolved_img.relative_to(resolved_input)
        if ".." in rel.parts:
            raise ValueError(f"Invalid path component '..': {rel}")
        sidecar = find_takeout_sidecar(img_path)
        dest_dir = filter_dir / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        if sidecar is not None and sidecar.exists():
            dest, sidecar_dest = unique_reject_dest_pair(dest_dir, img_path.name, sidecar)
        else:
            dest = unique_reject_path(dest_dir, img_path.name)
            sidecar_dest = None
        shutil.move(str(img_path), str(dest))
        if sidecar is not None and sidecar.exists() and sidecar_dest is not None:
            shutil.move(str(sidecar), str(sidecar_dest))
        return dest

    if move_lock is None:
        return _move()
    with move_lock:
        return _move()


def apply_from_report(
    report_path: Path,
    input_dir: Path,
    filter_dir: Path,
    *,
    threshold: float | None,
    threshold_ai: float | None,
    threshold_quality: float | None,
    threshold_generation: float | None,
    force_reapply: bool = False,
):
    meta, results = load_report(report_path)
    if meta is None:
        meta = {}
    thresholds = resolve_apply_thresholds(
        meta,
        threshold=threshold,
        threshold_ai=threshold_ai,
        threshold_quality=threshold_quality,
        threshold_generation=threshold_generation,
    )
    active = [f"{d}={thresholds[d]:.1f}" for d in SCORED_DIMENSIONS if thresholds.get(d) is not None]
    if active:
        print(f"Apply thresholds: {', '.join(active)}")
    moved = kept = skipped = already_applied = 0

    for entry in results:
        filename = entry["file"]
        if entry.get("applied") and not force_reapply:
            print(f"Skipping {filename}: already applied")
            already_applied += 1
            continue

        reject, reasons = evaluate_entry(entry, thresholds)
        if reject is None:
            print(f"Skipping {filename}: error entry without keep override")
            skipped += 1
            continue

        src = input_dir / filename
        if not src.exists():
            print(f"Warning: {filename} not found, skipping")
            skipped += 1
            continue

        reason_text = ", ".join(reasons)
        if reject:
            try:
                dest = move_reject(src, input_dir, filter_dir)
                try:
                    dest_print = str(dest.relative_to(filter_dir))
                except ValueError:
                    dest_print = dest.name
                print(f"Moved {filename} → {dest_print} ({reason_text})")
                entry["applied"] = True
                entry["applied_at"] = datetime.now(timezone.utc).isoformat()
                moved += 1
            except (OSError, ValueError) as e:
                print(f"Error moving {filename}: {e}")
                skipped += 1
        else:
            print(f"Preserved {filename} ({reason_text})")
            kept += 1

    meta["last_applied_at"] = datetime.now(timezone.utc).isoformat()
    meta["last_applied_thresholds"] = thresholds
    write_report(report_path, meta, results)

    parts = [f"{moved} moved", f"{kept} kept", f"{skipped} skipped"]
    if already_applied:
        parts.append(f"{already_applied} already applied")
    print(f"\nApply complete: {', '.join(parts)}.")


def scaled_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int] | None:
    if max_dimension <= 0:
        return None
    long_edge = max(width, height)
    if long_edge <= max_dimension:
        return None
    scale = max_dimension / long_edge
    return max(1, round(width * scale)), max(1, round(height * scale))


def get_original_stem(stem: str) -> str | None:
    original = stem
    
    m = re.match(r'^(.*?)[-_]edited$', original, re.IGNORECASE)
    if m:
        original = m.group(1)
        
    m = re.match(r'^IMG_E(.*)$', original, re.IGNORECASE)
    if m:
        original = f"IMG_{m.group(1)}"
        
    return original if original != stem else None


def build_edit_policy_map(image_paths: list[Path], input_dir: Path, policy: str) -> dict[str, str]:
    if policy == "both":
        return {}
        
    edit_rejects: dict[str, str] = {}
    
    dir_map: dict[Path, list[Path]] = {}
    for p in image_paths:
        dir_map.setdefault(p.parent, []).append(p)
        
    for paths in dir_map.values():
        stem_map: dict[str, list[Path]] = {}
        for p in paths:
            stem_map.setdefault(p.stem.lower(), []).append(p)
        
        for p in paths:
            orig_stem = get_original_stem(p.stem)
            if orig_stem and orig_stem.lower() in stem_map:
                edited_p = p
                candidates = stem_map[orig_stem.lower()]
                
                # Prioritize exact extension match
                exact_ext = [c for c in candidates if c.suffix.lower() == edited_p.suffix.lower()]
                targets = exact_ext if exact_ext else candidates
                
                for orig_p in targets:
                    if policy == "original":
                        edit_rejects[edited_p.relative_to(input_dir).as_posix()] = "edit_policy (prefer original)"
                    elif policy == "edited":
                        edit_rejects[orig_p.relative_to(input_dir).as_posix()] = "edit_policy (prefer edited)"

    return edit_rejects


def build_dupe_map(image_paths: list[Path], input_dir: Path) -> dict[str, str]:
    """Map duplicate relative path -> keeper relative path (first by sorted path)."""
    by_hash: dict[str, str] = {}
    dupes: dict[str, str] = {}
    for path in sorted(image_paths, key=lambda p: str(p.relative_to(input_dir))):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel_path = path.relative_to(input_dir).as_posix()
        if digest in by_hash:
            dupes[rel_path] = by_hash[digest]
        else:
            by_hash[digest] = rel_path
    return dupes


TAKEOUT_SIDECAR_MAX_LEN = 51
TAKEOUT_EXIF_SUFFIXES = frozenset({".jpg", ".jpeg"})
_DUP_MARKER_RE = re.compile(r"^(.+)\((\d+)\)(\.[^.]+)$")
_PROTECTED_JSON_STEMS = frozenset(Path(name).stem for name in (*LEGACY_REPORT_NAMES, DEFAULT_REPORT_NAME))


def _fit_takeout_sidecar(base: str) -> str:
    full = f"{base}.json"
    if len(full) <= TAKEOUT_SIDECAR_MAX_LEN:
        return full
    return f"{base[: TAKEOUT_SIDECAR_MAX_LEN - len('.json')]}.json"


def takeout_sidecar_candidate_names(name: str) -> list[str]:
    """Candidate Google Takeout sidecar filenames for a media basename."""
    names = [
        _fit_takeout_sidecar(name),
        _fit_takeout_sidecar(f"{name}.supplemental-metadata"),
    ]
    m = _DUP_MARKER_RE.match(name)
    if m:
        bare = f"{m.group(1)}{m.group(3)}"
        num = m.group(2)
        full_base = f"{bare}.supplemental-metadata"
        full_with_json = f"{full_base}.json"
        trunc_base = (
            full_base
            if len(full_with_json) <= TAKEOUT_SIDECAR_MAX_LEN
            else full_base[: TAKEOUT_SIDECAR_MAX_LEN - len(".json")]
        )
        names.append(f"{trunc_base}({num}).json")
        names.append(f"{bare}({num}).json")
    return names


def _bracket_swap_name(filename: str) -> str | None:
    matches = list(re.finditer(r"\(\d+\)\.", filename))
    if not matches:
        return None
    match = matches[-1]
    bracket = match.group(0)[:-1]
    idx = filename.rfind(bracket)
    if idx < 0:
        return None
    without = filename[:idx] + filename[idx + len(bracket) :]
    return f"{without}{bracket}"


def _takeout_basename_variants(filename: str) -> list[str]:
    variants = [filename]
    swapped = _bracket_swap_name(filename)
    if swapped is not None:
        variants.append(swapped)
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    if suffix and stem not in _PROTECTED_JSON_STEMS:
        variants.append(stem)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in variants:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def find_takeout_sidecar(img_path: Path) -> Path | None:
    """Locate a Google Takeout JSON sidecar for an image, if present."""
    directory = img_path.parent
    for variant in _takeout_basename_variants(img_path.name):
        for candidate in takeout_sidecar_candidate_names(variant):
            sidecar = directory / candidate
            if sidecar.is_file():
                return sidecar
    return None


def parse_takeout_sidecar(path: Path) -> dict:
    """Extract creation timestamp and GPS from a Takeout sidecar JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict = {}
    for key in ("photoTakenTime", "creationTime"):
        block = data.get(key)
        if isinstance(block, dict) and block.get("timestamp") is not None:
            try:
                result["timestamp"] = int(str(block["timestamp"]))
                break
            except ValueError:
                continue
    for geo_key in ("geoDataExif", "geoData"):
        geo = data.get(geo_key)
        if not isinstance(geo, dict):
            continue
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        if lat is None or lon is None:
            continue
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if lat_f == 0.0 and lon_f == 0.0:
            continue
        result["latitude"] = lat_f
        result["longitude"] = lon_f
        alt = geo.get("altitude")
        if alt is not None:
            try:
                altitude = float(alt)
            except (TypeError, ValueError):
                pass
            else:
                if math.isfinite(altitude):
                    result["altitude"] = altitude
        break
    return result


def _deg_to_dms_rational(deg: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    deg = abs(deg)
    d = int(deg)
    minutes_float = (deg - d) * 60
    m = int(minutes_float)
    s = round((minutes_float - m) * 60 * 10_000)
    if s >= 600_000:
        s -= 600_000
        m += 1
    if m == 60:
        m = 0
        d += 1
    return (d, 1), (m, 1), (s, 10_000)


def merge_takeout_metadata(img_path: Path, sidecar_path: Path | None = None) -> bool:
    """Write missing DateTimeOriginal/GPS from a Takeout sidecar into JPEG EXIF."""
    if img_path.suffix.lower() not in TAKEOUT_EXIF_SUFFIXES:
        return False
    sidecar = sidecar_path or find_takeout_sidecar(img_path)
    if sidecar is None:
        return False
    meta = parse_takeout_sidecar(sidecar)
    if not meta:
        return False

    import piexif

    try:
        exif_dict = piexif.load(str(img_path))
    except (piexif.InvalidImageDataError, ValueError, OSError):
        return False

    modified = False
    if "timestamp" in meta:
        dt_str = datetime.fromtimestamp(meta["timestamp"]).strftime("%Y:%m:%d %H:%M:%S")  # noqa: DTZ006
        zeroth = exif_dict.setdefault("0th", {})
        exif_ifd = exif_dict.setdefault("Exif", {})
        if not zeroth.get(piexif.ImageIFD.DateTime):
            zeroth[piexif.ImageIFD.DateTime] = dt_str
            modified = True
        if not exif_ifd.get(piexif.ExifIFD.DateTimeOriginal):
            exif_ifd[piexif.ExifIFD.DateTimeOriginal] = dt_str
            modified = True
        if not exif_ifd.get(piexif.ExifIFD.DateTimeDigitized):
            exif_ifd[piexif.ExifIFD.DateTimeDigitized] = dt_str
            modified = True

    if "latitude" in meta and "longitude" in meta:
        lat, lon = meta["latitude"], meta["longitude"]
        gps_ifd = exif_dict.setdefault("GPS", {})
        if not gps_ifd.get(piexif.GPSIFD.GPSLatitude):
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
            gps_ifd[piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(lat)
            modified = True
        if not gps_ifd.get(piexif.GPSIFD.GPSLongitude):
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
            gps_ifd[piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(lon)
            modified = True
        if "altitude" in meta and not gps_ifd.get(piexif.GPSIFD.GPSAltitude):
            alt = meta["altitude"]
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
            gps_ifd[piexif.GPSIFD.GPSAltitude] = (round(abs(alt) * 100), 100)
            modified = True

    if modified:
        piexif.insert(piexif.dump(exif_dict), str(img_path))
    return modified


def remove_takeout_sidecar(img_path: Path, *, sidecar_path: Path | None = None) -> bool:
    sidecar = sidecar_path or find_takeout_sidecar(img_path)
    if sidecar is None:
        return False
    try:
        sidecar.unlink()
    except OSError:
        return False
    return True


def _is_solid_color(img) -> bool:
    # ponytail: 8x8 downsample; per-channel range <= 2 catches blank/solid renders; upgrade: histogram
    small = img.resize((8, 8))
    pixels = list(small.getdata())
    if len(pixels) <= 1:
        return True
    channels = len(pixels[0]) if isinstance(pixels[0], tuple) else 1
    for ch in range(channels):
        values = [p[ch] if isinstance(p, tuple) else p for p in pixels]
        if max(values) - min(values) > 2:
            return False
    return True


def check_hygiene(
    img_path: Path,
    *,
    dupe_of: str | None = None,
    min_res: tuple[int, int] | None = None,
    edit_reject_reason: str | None = None,
) -> dict:
    if dupe_of is not None:
        return {"action": "reject", "exact_dupe_of": dupe_of}
    if edit_reject_reason is not None:
        return {"action": "reject", "reason": edit_reject_reason}

    from PIL import Image, ImageOps

    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            width, height = img.size
            if min_res is not None and (width < min_res[0] or height < min_res[1]):
                return {"action": "reject", "reason": f"below_min_res ({width}x{height})"}
            if _is_solid_color(img):
                return {"action": "reject", "reason": "solid_color"}
    except Exception:  # noqa: BLE001 — any decode failure ⇒ corrupt
        return {"action": "reject", "reason": "corrupt"}

    return {"action": "keep"}


def prepare_analysis_image(img_path: Path, max_dimension: int) -> tuple[Path, Path | None]:
    """Return (path for Ollama, temp file to delete after analysis, or None)."""
    suffix = img_path.suffix.lower()
    transcode_heif = suffix in HEIF_SUFFIXES
    if max_dimension <= 0 and not transcode_heif:
        return img_path, None
    from PIL import Image, ImageOps

    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img)
        new_size = scaled_dimensions(*img.size, max_dimension) if max_dimension > 0 else None
        if new_size is None and not transcode_heif:
            return img_path, None
        if new_size is not None:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        out_suffix = ".jpg" if transcode_heif else suffix
        fd, temp_name = tempfile.mkstemp(suffix=out_suffix)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            save_kwargs = {}
            if out_suffix in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
            if transcode_heif and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(temp_path, **save_kwargs)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return temp_path, temp_path


def ensure_model(model_name: str):
    print(f"Checking if model '{model_name}' is available locally...")
    try:
        ollama.show(model_name)
        print(f"Model '{model_name}' is ready.")
    except ollama.ResponseError as e:
        if e.status_code == 404:
            print(f"Model '{model_name}' not found. Pulling... (this may take a while)")
            ollama.pull(model_name)
            print(f"Successfully pulled '{model_name}'.")
        else:
            raise SystemExit(f"Error: Ollama returned status {e.status_code} for model '{model_name}': {e}")
    except Exception as e:  # noqa: BLE001 — connection / non-ResponseError from client
        raise SystemExit(f"Error: Cannot reach Ollama. Is 'ollama serve' running? {e}")


def process_image(
    img_path: Path,
    input_dir: Path,
    model_name: str,
    config: CullConfig,
    cli_threshold: float,
    dry_run: bool,
    filter_dir: Path,
    max_dimension: int,
    fast: bool,
    print_lock: threading.Lock,
    move_lock: threading.Lock,
    progress: ScanProgress | None = None,
    *,
    dupe_of: str | None = None,
    min_res: tuple[int, int] | None = None,
    edit_reject_reason: str | None = None,
    takeout_merged: bool = False,
) -> dict:
    tag = f"{progress.begin()} " if progress else ""
    rel_path = img_path.relative_to(input_dir).as_posix()

    with print_lock:
        print(f"{tag}Analyzing {rel_path}...")

    blocks: dict[str, dict] = {}
    try:
        if "hygiene" in config.lenses:
            hygiene_dict = check_hygiene(img_path, dupe_of=dupe_of, min_res=min_res, edit_reject_reason=edit_reject_reason)
            HygieneVerdict(**hygiene_dict)
            blocks["hygiene"] = hygiene_dict
            with print_lock:
                if hygiene_dict["action"] == "reject":
                    detail = hygiene_dict.get("exact_dupe_of") or hygiene_dict.get("reason") or "reject"
                    print(f"{tag}  Hygiene: reject ({detail})")
                else:
                    print(f"{tag}  Hygiene: keep")
            if hygiene_dict["action"] == "reject":
                if config.multi_dimensional:
                    result = result_entry_from_lenses(rel_path, **blocks)
                else:
                    result = {"file": rel_path, "hygiene": hygiene_dict}
                if not dry_run:
                    meta = {"threshold": cli_threshold, "thresholds": config.thresholds}
                    if should_reject(result, cli_threshold, meta):
                        try:
                            dest = move_reject(img_path, input_dir, filter_dir, move_lock)
                            with print_lock:
                                print(f"{tag}  -> Moved filtered image to {dest}")
                        except (OSError, ValueError) as e:
                            with print_lock:
                                print(f"{tag}  -> Error moving {rel_path}: {e}")
                            result["move_error"] = str(e)
                    else:
                        with print_lock:
                            print(f"{tag}  -> Preserved keeper in place")
                        if not dry_run and takeout_merged:
                            remove_takeout_sidecar(img_path)
                return result
        if "ai" in config.lenses:
            analysis_dict = analyze_image(img_path, model_name, max_dimension, fast)
            analysis = RealismAnalysis(**analysis_dict)
            blocks["ai"] = analysis_to_ai_block(analysis_dict)
            with print_lock:
                print(f"{tag}  Score: {analysis.realism_score} - Realistic: {analysis.is_realistic}")
        if "generation" in config.lenses:
            gen_dict = analyze_generation(img_path, model_name, max_dimension, fast)
            verdict = GenerationVerdict(**gen_dict)
            blocks["generation"] = {
                "success_score": verdict.success_score,
                "issues": verdict.issues,
                "reasoning": verdict.reasoning,
            }
            with print_lock:
                print(f"{tag}  Success: {verdict.success_score}")
        if "quality" in config.lenses:
            qual_dict = analyze_quality(img_path, model_name, max_dimension, fast)
            verdict = QualityVerdict(**qual_dict)
            blocks["quality"] = {
                "keeper_score": verdict.keeper_score,
                "issues": verdict.issues,
                "reasoning": verdict.reasoning,
            }
            with print_lock:
                print(f"{tag}  Keeper: {verdict.keeper_score}")
    except Exception as e:  # noqa: BLE001 — per-image: log and continue batch
        with print_lock:
            traceback.print_exc()
            print(f"{tag}Error processing {rel_path}: {e}")
        return {"file": rel_path, "status": "error", "error": str(e)}

    if config.multi_dimensional:
        result = result_entry_from_lenses(rel_path, **blocks)
    elif blocks.get("ai") is not None:
        # Legacy single-score: rebuild analysis dict from ai block for compat
        ai = blocks["ai"]
        result = realism_result_entry(
            rel_path,
            {
                "realism_score": ai["realism_score"],
                "is_realistic": ai.get("is_realistic", False),
                "detected_artifacts": ai.get("issues", []),
                "reasoning": ai.get("reasoning", ""),
            },
        )
    else:
        result = {"file": rel_path}

    if not dry_run:
        meta = {"threshold": cli_threshold, "thresholds": config.thresholds}
        if should_reject(result, cli_threshold, meta):
            try:
                dest = move_reject(img_path, input_dir, filter_dir, move_lock)
                with print_lock:
                    print(f"{tag}  -> Moved filtered image to {dest}")
            except (OSError, ValueError) as e:
                with print_lock:
                    print(f"{tag}  -> Error moving {rel_path}: {e}")
                result["move_error"] = str(e)
        else:
            with print_lock:
                print(f"{tag}  -> Preserved keeper in place")
            if not dry_run and takeout_merged:
                remove_takeout_sidecar(img_path)

    return result


def analyze_quality(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
    send_path, temp_path = prepare_analysis_image(img_path, max_dimension)
    if fast:
        prompt = (
            "Score this real photograph for album keeper quality (1.0-10.0). "
            "Would someone keep this in a photo album? Focus on blur, exposure, "
            "framing accidents, and screenshots — NOT whether the image is AI-generated. "
            "List issues as short snake_case tags (e.g. motion_blur, underexposed, "
            "eyes_closed, screenshot). Return keeper_score and issues only. Do not explain."
        )
        schema = QualityVerdictFast.model_json_schema()
    else:
        prompt = (
            "Score this real photograph for KEEPER / album-worthiness (1.0-10.0): "
            "would someone keep this in a photo album? Focus on technical and compositional "
            "quality — NOT whether the image is AI-generated or synthetic. "
            "Flag concrete issues as short snake_case tags, e.g. motion_blur, out_of_focus, "
            "underexposed, overexposed, eyes_closed, accidental_frame, finger_on_lens, "
            "pocket_shot, floor_shot, screenshot, ui_chrome, duplicate_feel. "
            "High scores for sharp, well-exposed, intentional shots; low scores for "
            "accidental pocket/floor captures, heavy blur, closed eyes on faces, or "
            "screenshots with UI chrome. Explain your reasoning concisely (1-2 short sentences max)."
        )
        schema = QualityVerdict.model_json_schema()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(send_path)],
            }],
            format=schema,
            options={"temperature": 0},
        )
        try:
            result = json.loads(response.message.content)
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeError(f"Model returned invalid JSON (likely truncated): {e}") from e
            
        if fast:
            QualityVerdictFast(**result)
            result["reasoning"] = ""
        else:
            QualityVerdict(**result)
        return result
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def analyze_generation(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
    send_path, temp_path = prepare_analysis_image(img_path, max_dimension)
    if fast:
        prompt = (
            "Score this AI-generated image for generation success (1.0-10.0). "
            "Did the render succeed? Focus on subject coherence, anatomy sanity, and generation "
            "artifacts — NOT photorealism. Do not penalize stylization or surreal mashups. "
            "Return success_score and issues only. Do not explain."
        )
        schema = GenerationVerdictFast.model_json_schema()
    else:
        prompt = (
            "Score this AI-generated image for generation SUCCESS (1.0-10.0): did the render "
            "succeed and is it worth keeping? Focus on subject coherence, anatomical/proportion "
            "sanity for depicted subjects, artifact severity, and composition readability. "
            "Do NOT penalize stylization, surreal mashups, cartoon aesthetics, or non-photoreal "
            "intent — a coherent creative mashup scores high. Flag hard failures: extra/missing "
            "limbs, melted features, garbled text, unreadable subjects, obvious generation "
            "collapse. List issues as short snake_case tags (e.g. extra_limbs, garbled_text, "
            "subject_unrecognizable). Explain your reasoning concisely (1-2 short sentences max)."
        )
        schema = GenerationVerdict.model_json_schema()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(send_path)],
            }],
            format=schema,
            options={"temperature": 0},
        )
        try:
            result = json.loads(response.message.content)
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeError(f"Model returned invalid JSON (likely truncated): {e}") from e
            
        if fast:
            GenerationVerdictFast(**result)
            result["reasoning"] = ""
        else:
            GenerationVerdict(**result)
        return result
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def analyze_image(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
    send_path, temp_path = prepare_analysis_image(img_path, max_dimension)
    if fast:
        prompt = (
            "Analyze this image for photorealism. Return realism_score (1.0-10.0), "
            "is_realistic, and detected_artifacts only. Do not explain."
        )
        schema = RealismAnalysisFast.model_json_schema()
    else:
        prompt = (
            "Analyze this image for photorealism. Identify if it is realistic, rate it from 1.0 to 10.0, "
            "list any AI-generated artifacts, and explain your reasoning concisely (1-2 short sentences max)."
        )
        schema = RealismAnalysis.model_json_schema()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(send_path)]
            }],
            format=schema,
            options={"temperature": 0}
        )
        try:
            result = json.loads(response.message.content)
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeError(f"Model returned invalid JSON (likely truncated): {e}") from e
            
        if fast:
            RealismAnalysisFast(**result)
            result["reasoning"] = ""
        return result
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run_cull(args, input_dir: Path, filter_dir: Path):
    config = resolve_cull_config(
        profile=args.profile,
        checks=args.checks,
        threshold=args.threshold,
        threshold_ai=args.threshold_ai,
        threshold_quality=args.threshold_quality,
        threshold_generation=args.threshold_generation,
    )
    validate_cull_config(config)

    needs_vlm = any(lens in SCORED_DIMENSIONS for lens in config.lenses)
    if needs_vlm:
        ensure_model(args.model)

    if getattr(args, "recursive", False):
        filter_root = filter_dir.resolve(strict=False)
        image_paths = [
            p
            for p in input_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
            and p.is_file()
            and not p.resolve(strict=False).is_relative_to(filter_root)
        ]
    else:
        image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()]

    edit_policy = getattr(args, "edit_policy", None) or "original"
    edit_rejects = build_edit_policy_map(image_paths, input_dir, edit_policy) if "hygiene" in config.lenses else {}
    if "hygiene" in config.lenses:
        dupe_paths = [p for p in image_paths if p.relative_to(input_dir).as_posix() not in edit_rejects]
        dupe_map = build_dupe_map(dupe_paths, input_dir)
    else:
        dupe_map = {}
    if getattr(args, "edit_policy", None) is not None and "hygiene" not in config.lenses:
        print(f"Warning: --edit-policy ignored because hygiene lens is not active (lenses: {', '.join(config.lenses)})")
    if args.min_res is not None and "hygiene" not in config.lenses:
        print(f"Warning: --min-res ignored because hygiene lens is not active (lenses: {', '.join(config.lenses)})")
    min_res = args.min_res if "hygiene" in config.lenses else None

    takeout_merged: set[str] = set()
    if not args.dry_run:
        merged = 0
        for img_path in image_paths:
            rel = img_path.relative_to(input_dir).as_posix()
            try:
                if merge_takeout_metadata(img_path):
                    takeout_merged.add(rel)
                    merged += 1
            except Exception as e:  # noqa: BLE001 — per-image: log and continue batch
                print(f"Warning: Takeout metadata merge failed for {rel}: {e}")
        if merged:
            print(f"Merged Takeout sidecar metadata into {merged} JPEG(s).")

    depth = pipeline_depth(len(image_paths))
    print_lock = threading.Lock()
    move_lock = threading.Lock()
    progress = ScanProgress(len(image_paths)) if image_paths else None

    if config.profile:
        print(f"Profile: {config.profile} (lenses: {', '.join(config.lenses)})")

    def process(img_path: Path) -> dict:
        return process_image(
            img_path,
            input_dir,
            args.model,
            config,
            args.threshold,
            args.dry_run,
            filter_dir,
            args.max_dimension,
            args.fast,
            print_lock,
            move_lock,
            progress,
            dupe_of=dupe_map.get(img_path.relative_to(input_dir).as_posix()),
            min_res=min_res,
            edit_reject_reason=edit_rejects.get(img_path.relative_to(input_dir).as_posix()),
            takeout_merged=img_path.relative_to(input_dir).as_posix() in takeout_merged,
        )

    with ThreadPoolExecutor(max_workers=depth) as executor:
        results = list(executor.map(process, image_paths))

    results.sort(key=score_sort_key)

    if results:
        report_path = input_dir / DEFAULT_REPORT_NAME
        meta = build_report_meta(
            threshold=args.threshold,
            model=args.model,
            max_dimension=args.max_dimension,
            fast=args.fast,
            dry_run=args.dry_run,
            profile=config.profile,
            thresholds=config.thresholds,
            min_res=min_res,
        )
        write_report(report_path, meta, results)
        display_report_path = args.report_path_display or str(report_path)
        print(f"\nCull complete! {len(results)} of {len(image_paths)} images in report.")
        print(f"Report saved to {display_report_path}")
    else:
        print(f"\nNo images were processed out of {len(image_paths)} found.")


def _self_check():
    assert should_reject({"file": "a.jpg", "keep": True, "analysis": {"realism_score": 1.0}}, 7.0) is False
    assert should_reject({"file": "b.jpg", "keep": False, "analysis": {"realism_score": 10.0}}, 7.0) is True
    assert should_reject({"file": "c.jpg", "analysis": {"realism_score": 6.0}}, 7.0) is True
    assert should_reject({"file": "d.jpg", "analysis": {"realism_score": 8.0}}, 7.0) is False
    assert should_reject({"file": "e.jpg", "status": "error", "error": "boom"}, 7.0) is None
    assert should_reject({"file": "f.jpg", "status": "error", "error": "boom", "keep": False}, 7.0) is True
    assert should_reject({"file": "g.jpg", "hygiene": {"action": "reject", "exact_dupe_of": "orig.jpg"}}, 7.0) is True
    assert should_reject({"file": "h.jpg", "hygiene": {"action": "keep"}}, 7.0) is False
    assert should_reject(
        {"file": "hk.jpg", "hygiene": {"action": "keep"}, "ai": {"realism_score": 5.0, "issues": [], "reasoning": ""}},
        7.0,
    ) is True
    assert should_reject(
        {"file": "i.jpg", "ai": {"realism_score": 6.0, "issues": [], "reasoning": ""}},
        7.0,
    ) is True
    assert should_reject(
        {"file": "j.jpg", "ai": {"realism_score": 8.0, "issues": [], "reasoning": ""}},
        7.0,
        {"thresholds": {"ai": 7.0, "quality": None, "generation": None}},
    ) is False
    assert should_reject({"file": "k.jpg", "quality": {"keeper_score": 3.0, "issues": []}}, 7.0) is None
    photos_meta = {"thresholds": {"ai": None, "quality": 6.0, "generation": None}}
    assert should_reject(
        {"file": "blur.jpg", "quality": {"keeper_score": 3.0, "issues": ["motion_blur"], "reasoning": ""}},
        6.0,
        photos_meta,
    ) is True
    assert should_reject(
        {"file": "sharp.jpg", "quality": {"keeper_score": 8.0, "issues": [], "reasoning": ""}},
        6.0,
        photos_meta,
    ) is False
    gen_meta = {"thresholds": {"ai": None, "quality": None, "generation": 7.0}}
    assert should_reject(
        {"file": "octo.jpg", "generation": {"success_score": 8.5, "issues": [], "reasoning": ""}},
        7.0,
        gen_meta,
    ) is False
    assert should_reject(
        {"file": "trex.jpg", "generation": {"success_score": 4.0, "issues": ["extra_limbs"], "reasoning": ""}},
        7.0,
        gen_meta,
    ) is True
    multi_entry = {
        "file": "both.jpg",
        "ai": {"realism_score": 8.0, "issues": [], "reasoning": ""},
        "quality": {"keeper_score": 3.0, "issues": ["motion_blur"], "reasoning": ""},
    }
    multi_thresh = {"ai": 7.0, "quality": 6.0, "generation": None}
    reject, reasons = evaluate_entry(multi_entry, multi_thresh)
    assert reject is True and any("quality:" in r for r in reasons) and any("motion_blur" in r for r in reasons)
    reject, reasons = evaluate_entry(
        {"file": "octo.jpg", "generation": {"success_score": 8.5, "issues": [], "reasoning": ""}},
        {"ai": None, "quality": None, "generation": 7.0},
    )
    assert reject is False and reasons == ["generation: success_score 8.5"]
    reject, reasons = evaluate_entry(
        {"file": "x.jpg", "keep": True, "quality": {"keeper_score": 1.0, "issues": []}},
        {"ai": None, "quality": 6.0, "generation": None},
    )
    assert reject is False and reasons == ["keep: true override"]
    resolved = resolve_apply_thresholds(
        {"thresholds": {"ai": None, "quality": 6.0, "generation": None}},
        threshold=7.5,
        threshold_ai=None,
        threshold_quality=None,
        threshold_generation=None,
    )
    assert resolved["quality"] == 7.5 and resolved["ai"] is None
    resolved = resolve_apply_thresholds(
        {"profile": "mixed", "thresholds": {"ai": 7.0, "quality": 6.0, "generation": None}},
        threshold=8.0,
        threshold_ai=None,
        threshold_quality=None,
        threshold_generation=None,
    )
    assert resolved["ai"] == 8.0 and resolved["quality"] == 6.0
    resolved = resolve_apply_thresholds(
        {"thresholds": {"ai": 7.0, "quality": 6.0, "generation": None}},
        threshold=None,
        threshold_ai=8.5,
        threshold_quality=None,
        threshold_generation=None,
    )
    assert resolved["ai"] == 8.5 and resolved["quality"] == 6.0
    with tempfile.TemporaryDirectory() as tmp:
        import io
        from contextlib import redirect_stdout

        input_dir = Path(tmp)
        report = input_dir / "realism_audit_report.json"
        write_report(
            report,
            {"threshold": 7.0, "thresholds": {"ai": 7.0, "quality": None, "generation": None}},
            [{"file": "bad.jpg", "analysis": {"realism_score": 4.0, "is_realistic": False, "detected_artifacts": [], "reasoning": ""}}],
        )
        (input_dir / "bad.jpg").write_bytes(b"x")
        assert resolve_report_path(input_dir, None) == report
        captured = io.StringIO()
        with redirect_stdout(captured):
            apply_from_report(
                report, input_dir, input_dir / "rejects",
                threshold=None, threshold_ai=None, threshold_quality=None, threshold_generation=None,
            )
        out = captured.getvalue()
        assert "Moved bad.jpg" in out and "realism_score 4.0 < 7.0" in out
        assert not (input_dir / "bad.jpg").exists()
        meta_after, results_after = load_report(report)
        assert results_after[0]["applied"] is True and results_after[0]["applied_at"]
        assert meta_after["last_applied_at"] and meta_after["last_applied_thresholds"]["ai"] == 7.0
        captured = io.StringIO()
        with redirect_stdout(captured):
            apply_from_report(
                report, input_dir, input_dir / "rejects",
                threshold=None, threshold_ai=None, threshold_quality=None, threshold_generation=None,
            )
        assert "already applied" in captured.getvalue()
        (input_dir / "bad.jpg").write_bytes(b"x")
        captured = io.StringIO()
        with redirect_stdout(captured):
            apply_from_report(
                report, input_dir, input_dir / "rejects",
                threshold=None, threshold_ai=None, threshold_quality=None, threshold_generation=None,
                force_reapply=True,
            )
        assert "Moved bad.jpg" in captured.getvalue()
    with tempfile.TemporaryDirectory() as tmp:
        import io
        from contextlib import redirect_stdout

        input_dir = Path(tmp)
        report = input_dir / "cull_report.json"
        write_report(
            report,
            {"profile": "photos", "thresholds": {"ai": None, "quality": 6.0, "generation": None}},
            [{"file": "blur.jpg", "quality": {"keeper_score": 3.1, "issues": ["motion_blur"], "reasoning": ""}}],
        )
        (input_dir / "blur.jpg").write_bytes(b"x")
        assert resolve_report_path(input_dir, None) == report
        captured = io.StringIO()
        with redirect_stdout(captured):
            apply_from_report(
                report, input_dir, input_dir / "rejects",
                threshold=None, threshold_ai=None, threshold_quality=None, threshold_generation=None,
            )
        out = captured.getvalue()
        assert "Moved blur.jpg" in out and "motion_blur" in out and "keeper_score 3.1 < 6.0" in out
    ai_fun = resolve_cull_config(
        profile="ai-fun", checks=None, threshold=7.0,
        threshold_ai=None, threshold_quality=None, threshold_generation=None,
    )
    assert ai_fun.profile == "ai-fun" and ai_fun.lenses == ("generation",)
    validate_cull_config(ai_fun)
    assert effective_thresholds({"threshold": 8.0, "thresholds": {"ai": None, "quality": 6.0, "generation": None}}, 7.0)["ai"] == 8.0
    assert multi_dimensional_active({"ai": 7.0, "quality": 6.0, "generation": None}) is True
    assert multi_dimensional_active({"ai": 7.0, "quality": None, "generation": None}) is False
    mixed = resolve_cull_config(profile="mixed", checks=None, threshold=7.5, threshold_ai=None, threshold_quality=None, threshold_generation=None)
    assert mixed.profile == "mixed" and mixed.lenses == ("hygiene", "ai") and mixed.multi_dimensional
    assert mixed.thresholds["ai"] == 7.5 and mixed.thresholds["quality"] == 6.0
    photos_ai = resolve_cull_config(profile="photos", checks=("ai",), threshold=8.0, threshold_ai=None, threshold_quality=None, threshold_generation=None)
    assert photos_ai.thresholds["ai"] == 8.0 and photos_ai.thresholds["quality"] is None
    legacy = resolve_cull_config(profile=None, checks=None, threshold=8.0, threshold_ai=None, threshold_quality=None, threshold_generation=None)
    assert legacy.profile is None and not legacy.multi_dimensional and legacy.thresholds["ai"] == 8.0
    assert parse_checks("hygiene,ai") == ("hygiene", "ai")
    assert parse_min_res("512x512") == (512, 512)
    assert parse_min_res(None) is None
    try:
        parse_checks("")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty --checks")
    photos = resolve_cull_config(profile="photos", checks=None, threshold=6.0, threshold_ai=None, threshold_quality=None, threshold_generation=None)
    assert photos.profile == "photos" and photos.lenses == ("hygiene", "quality") and photos.thresholds["quality"] == 6.0
    validate_cull_config(photos)
    ai_block = analysis_to_ai_block(
        {"realism_score": 8.0, "is_realistic": True, "detected_artifacts": ["x"], "reasoning": "ok"}
    )
    assert ai_block["issues"] == ["x"] and ai_block["realism_score"] == 8.0
    legacy_entry = realism_result_entry("a.jpg", {"realism_score": 5.0, "is_realistic": False, "detected_artifacts": [], "reasoning": ""})
    assert "analysis" in legacy_entry and "ai" not in legacy_entry
    multi_entry = realism_result_entry(
        "b.jpg",
        {"realism_score": 5.0, "is_realistic": False, "detected_artifacts": [], "reasoning": ""},
        multi_dimensional=True,
    )
    assert "ai" in multi_entry and "analysis" not in multi_entry
    assert score_sort_key({"file": "z.jpg", "quality": {"keeper_score": 9.0}})[0] == -9.0
    assert score_sort_key({"file": "g.jpg", "generation": {"success_score": 9.0}})[0] == -9.0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump([{"file": "x.jpg", "analysis": {"realism_score": 5.0}}], f)
        f.flush()
        meta, results = load_report(Path(f.name))
    assert meta is None and results[0]["file"] == "x.jpg"
    assert pipeline_depth(0) == 1
    assert pipeline_depth(3) == 3
    assert pipeline_depth(100) == MAX_IN_FLIGHT
    assert scaled_dimensions(100, 50, 0) is None
    assert scaled_dimensions(100, 50, 200) is None
    assert scaled_dimensions(2000, 1000, 1024) == (1024, 512)
    assert scaled_dimensions(1000, 2000, 1024) == (512, 1024)
    assert scaled_dimensions(1, 2, 1) == (1, 1)
    assert scaled_dimensions(2, 1, 1) == (1, 1)
    assert "reasoning" not in RealismAnalysisFast.model_json_schema()["properties"]
    assert "reasoning" in RealismAnalysis.model_json_schema()["properties"]
    assert "reasoning" not in GenerationVerdictFast.model_json_schema()["properties"]
    assert "reasoning" in GenerationVerdict.model_json_schema()["properties"]
    assert "reasoning" not in QualityVerdictFast.model_json_schema()["properties"]
    assert "reasoning" in QualityVerdict.model_json_schema()["properties"]
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "big.png"
        Image.new("RGB", (400, 200), "red").save(src)
        send_path, temp_path = prepare_analysis_image(src, 100)
        assert temp_path is not None
        try:
            assert send_path != src
            assert Image.open(send_path).size == (100, 50)
        finally:
            temp_path.unlink(missing_ok=True)
        assert prepare_analysis_image(src, 0) == (src, None)
        assert prepare_analysis_image(src, 500) == (src, None)
    with tempfile.TemporaryDirectory() as tmp:
        rejects = Path(tmp)
        (rejects / "a.jpg").write_text("old")
        assert unique_reject_path(rejects, "a.jpg") == rejects / "a_2.jpg"
        assert unique_reject_path(rejects, "b.jpg") == rejects / "b.jpg"
    p = ScanProgress(3)
    assert p.begin() == "[1/3]"
    assert p.begin() == "[2/3]"
    seen = []

    def _grab():
        seen.append(p.begin())

    threads = [threading.Thread(target=_grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 8 and len(set(seen)) == 8
    _check_process_image_error_handling()
    _check_run_cull_progress_tags()
    _check_generation_profile_cull()
    _check_quality_profile_cull()
    _check_hygiene_profile_cull()
    _check_recursive_cull()
    _check_quality_score_bounds()
    _check_quality_fast_issue_validation()
    _check_heif_support()
    _check_edit_policy()
    _check_takeout_metadata()


def _check_takeout_metadata():
    import piexif
    from PIL import Image

    sidecar_json = {
        "photoTakenTime": {"timestamp": "1453591193"},
        "geoDataExif": {"latitude": 40.7855556, "longitude": -73.9766667, "altitude": 44.0},
    }
    long_name = "A" * 40 + ".jpg"
    assert takeout_sidecar_candidate_names(long_name) == [
        _fit_takeout_sidecar(long_name),
        _fit_takeout_sidecar(f"{long_name}.supplemental-metadata"),
    ]
    truncated = takeout_sidecar_candidate_names("Screenshot_20231027_123303_Facebook.jpg")[1]
    assert truncated.endswith(".json") and len(truncated) == TAKEOUT_SIDECAR_MAX_LEN

    dup_names = takeout_sidecar_candidate_names("photo(3).jpg")
    assert "photo.jpg.supplemental-metadata(3).json" in dup_names
    assert "photo.jpg(3).json" in dup_names

    bracket = _takeout_basename_variants("image(11).jpg")
    assert "image.jpg(11)" in bracket

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_path = root / "20030616.jpg"
        Image.new("RGB", (8, 8), "red").save(img_path, format="JPEG")
        (root / "20030616.json").write_text(json.dumps(sidecar_json), encoding="utf-8")
        assert find_takeout_sidecar(img_path) == root / "20030616.json"

        img_path = root / "photo.jpg"
        Image.new("RGB", (8, 8), "red").save(img_path, format="JPEG")
        (root / "photo.jpg.supplemental-metadata.json").write_text(json.dumps(sidecar_json), encoding="utf-8")
        assert find_takeout_sidecar(img_path) == root / "photo.jpg.supplemental-metadata.json"
        assert merge_takeout_metadata(img_path) is True
        exif = piexif.load(str(img_path))
        expected_dt = datetime.fromtimestamp(1453591193).strftime("%Y:%m:%d %H:%M:%S").encode()  # noqa: DTZ006
        assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == expected_dt
        assert piexif.GPSIFD.GPSLatitude in exif["GPS"]

        img2_path = root / "has_gps.jpg"
        Image.new("RGB", (8, 8), "blue").save(img2_path, format="JPEG")
        existing = piexif.load(str(img2_path))
        existing.setdefault("GPS", {})
        existing["GPS"][piexif.GPSIFD.GPSLatitudeRef] = b"N"
        existing["GPS"][piexif.GPSIFD.GPSLatitude] = ((1, 1), (0, 1), (0, 1))
        existing["GPS"][piexif.GPSIFD.GPSLongitudeRef] = b"W"
        existing["GPS"][piexif.GPSIFD.GPSLongitude] = ((1, 1), (0, 1), (0, 1))
        piexif.insert(piexif.dump(existing), str(img2_path))
        (root / "has_gps.jpg.json").write_text(json.dumps(sidecar_json), encoding="utf-8")
        assert merge_takeout_metadata(img2_path) is True
        exif2 = piexif.load(str(img2_path))
        assert exif2["Exif"][piexif.ExifIFD.DateTimeOriginal] == expected_dt
        assert exif2["GPS"][piexif.GPSIFD.GPSLatitude] == existing["GPS"][piexif.GPSIFD.GPSLatitude]

        input_dir = root / "in"
        filter_dir = root / "out"
        input_dir.mkdir()
        filter_dir.mkdir()
        img3 = input_dir / "move_me.jpg"
        Image.new("RGB", (8, 8), "green").save(img3, format="JPEG")
        sidecar3 = input_dir / "move_me.jpg.json"
        sidecar3.write_text("{}", encoding="utf-8")
        move_reject(img3, input_dir, filter_dir)
        assert (filter_dir / "move_me.jpg").is_file()
        assert (filter_dir / "move_me.jpg.json").is_file()
        assert not sidecar3.exists()

        (filter_dir / "collision.jpg").write_bytes(b"x")
        (filter_dir / "collision_2.jpg.json").write_bytes(b"{}")
        img5 = input_dir / "collision.jpg"
        Image.new("RGB", (8, 8), "cyan").save(img5, format="JPEG")
        sidecar5 = input_dir / "collision.jpg.json"
        sidecar5.write_text("{}", encoding="utf-8")
        move_reject(img5, input_dir, filter_dir)
        assert (filter_dir / "collision_3.jpg").is_file()
        assert (filter_dir / "collision_3.jpg.json").is_file()

        img6 = input_dir / "20030616.jpg"
        Image.new("RGB", (8, 8), "magenta").save(img6, format="JPEG")
        sidecar6 = input_dir / "20030616.json"
        sidecar6.write_text("{}", encoding="utf-8")
        move_reject(img6, input_dir, filter_dir)
        assert (filter_dir / "20030616.jpg").is_file()
        assert (filter_dir / "20030616.json").is_file()

        (filter_dir / "image.jpg(11).json").write_bytes(b"{}")
        img7 = input_dir / "image(11).jpg"
        Image.new("RGB", (8, 8), "orange").save(img7, format="JPEG")
        (input_dir / "image.jpg(11).json").write_text("{}", encoding="utf-8")
        move_reject(img7, input_dir, filter_dir)
        assert (filter_dir / "image(11)_2.jpg").is_file()
        assert (filter_dir / "image.jpg(11)_2.json").is_file()

        bad_sidecar = root / "bad.jpg"
        Image.new("RGB", (8, 8), "white").save(bad_sidecar, format="JPEG")
        (root / "bad.jpg.json").write_text("{not json", encoding="utf-8")
        assert merge_takeout_metadata(bad_sidecar) is False

        img4 = input_dir / "keeper.jpg"
        Image.new("RGB", (8, 8), "yellow").save(img4, format="JPEG")
        sidecar4 = input_dir / "keeper.jpg.json"
        sidecar4.write_text(json.dumps(sidecar_json), encoding="utf-8")
        assert merge_takeout_metadata(img4) is True
        assert remove_takeout_sidecar(img4) is True
        assert not sidecar4.exists()


def _check_heif_support():
    from PIL import Image

    assert HEIF_SUFFIXES <= SUPPORTED_EXTENSIONS
    reg = Image.registered_extensions()
    assert reg.get(".heic") == "HEIF" and reg.get(".heif") == "HEIF"

    # Deterministic HEIF round-trip (no committed fixture; runs wherever _self_check runs).
    w, h = 64, 32
    pixels = bytes(c for y in range(h) for x in range(w) for c in (x * 4, y * 8, (x + y) % 256))
    with tempfile.TemporaryDirectory() as tmp:
        heif_path = Path(tmp) / "sample.heic"
        Image.frombytes("RGB", (w, h), pixels).save(heif_path, format="HEIF")
        with Image.open(heif_path) as im:
            im.load()
            assert im.size == (w, h)
        send_path, temp_path = prepare_analysis_image(heif_path, 0)
        assert temp_path is not None and send_path.suffix.lower() == ".jpg"
        try:
            with Image.open(send_path) as out:
                out.load()
                assert out.size == (w, h)
        finally:
            temp_path.unlink(missing_ok=True)


def _check_quality_score_bounds():
    from pydantic import ValidationError

    QualityVerdict(keeper_score=1.0, issues=[], reasoning="")
    QualityVerdict(keeper_score=10.0, issues=[], reasoning="")
    QualityVerdictFast(keeper_score=1.0, issues=[])
    QualityVerdictFast(keeper_score=10.0, issues=[])

    for bad in (0.9, 10.1, float("nan"), float("inf"), float("-inf")):
        for model, extra in ((QualityVerdict, {"reasoning": ""}), (QualityVerdictFast, {})):
            try:
                model(keeper_score=bad, issues=[], **extra)
                raise AssertionError(f"expected ValidationError for keeper_score={bad!r} in {model.__name__}")
            except ValidationError:
                pass


def _check_quality_fast_issue_validation():
    import sys
    from unittest.mock import MagicMock, patch

    from pydantic import ValidationError

    QualityVerdictFast(keeper_score=5.0, issues=["motion_blur", "underexposed"])
    try:
        QualityVerdictFast(keeper_score=5.0, issues=["Heavy motion blur"])
    except ValidationError:
        pass
    else:
        raise AssertionError("expected ValidationError for natural-language issue tag")

    mod = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "blur.jpg"
        img.write_bytes(b"x")
        mock_response = MagicMock()
        mock_response.message.content = json.dumps({
            "keeper_score": 4.0,
            "issues": ["very blurry image"],
        })
        with patch.object(mod, "prepare_analysis_image", return_value=(img, None)), patch(
            "ollama.chat", return_value=mock_response
        ):
            try:
                analyze_quality(img, "llava", fast=True)
                raise AssertionError("expected ValidationError for natural-language fast issues")
            except ValidationError:
                pass


def _check_quality_profile_cull():
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch

    from PIL import Image

    mod = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "photos"
        filter_dir = Path(tmp) / "rejects"
        input_dir.mkdir()
        for name, base in (("sunset.jpg", 200), ("pocket.jpg", 50), ("fail.jpg", 80)):
            img = Image.new("RGB", (800, 600))
            px = img.load()
            for y in range(600):
                for x in range(0, 800, 40):
                    px[x, y] = (base + x + y) % 256, base, (128 + y) % 256
            img.save(input_dir / name)

        def mock_quality(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
            if img_path.name == "fail.jpg":
                raise RuntimeError("boom")
            if img_path.name == "pocket.jpg":
                return {"keeper_score": 2.5, "issues": ["pocket_shot", "accidental_frame"], "reasoning": "accidental pocket capture"}
            return {"keeper_score": 8.5, "issues": [], "reasoning": "sharp sunset, well exposed"}

        config = resolve_cull_config(
            profile="photos", checks=None, threshold=6.0,
            threshold_ai=None, threshold_quality=None, threshold_generation=None,
        )
        captured = io.StringIO()
        with patch.object(mod, "analyze_quality", side_effect=mock_quality), redirect_stdout(captured), redirect_stderr(io.StringIO()):
            for img in sorted(input_dir.iterdir()):
                process_image(
                    img, input_dir, "llava", config, 6.0, True, filter_dir, 512, False,
                    threading.Lock(), threading.Lock(), ScanProgress(3),
                )

        out = captured.getvalue()
        assert "Hygiene: keep" in out
        assert "Keeper: 8.5" in out
        assert "Keeper: 2.5" in out
        assert "Error processing fail.jpg" in out

        args = argparse.Namespace(
            model="llava",
            threshold=6.0,
            threshold_ai=None,
            threshold_quality=None,
            threshold_generation=None,
            profile="photos",
            checks=None,
            dry_run=True,
            max_dimension=512,
            min_res=None,
            fast=False,
            report_path_display=None,
        )
        captured = io.StringIO()
        with patch.object(mod, "ensure_model"), patch.object(mod, "analyze_quality", side_effect=mock_quality), redirect_stdout(captured), redirect_stderr(io.StringIO()):
            run_cull(args, input_dir, filter_dir)

        meta, results = load_report(input_dir / DEFAULT_REPORT_NAME)
        assert meta["profile"] == "photos"
        assert meta["max_dimension"] == 512
        by_file = {r["file"]: r for r in results}
        assert by_file["sunset.jpg"]["quality"]["keeper_score"] == 8.5
        assert by_file["pocket.jpg"]["quality"]["keeper_score"] == 2.5
        assert "pocket_shot" in by_file["pocket.jpg"]["quality"]["issues"]
        assert by_file["fail.jpg"]["status"] == "error"


def _check_generation_profile_cull():
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch

    mod = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "photos"
        filter_dir = Path(tmp) / "rejects"
        input_dir.mkdir()
        for name in ("octo-rex.png", "six-finger-trex.png", "fail.png"):
            (input_dir / name).write_bytes(b"x")

        def mock_generation(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
            if img_path.name == "fail.png":
                raise RuntimeError("boom")
            if img_path.name == "six-finger-trex.png":
                return {"success_score": 3.0, "issues": ["extra_limbs"], "reasoning": "six fingers"}
            return {"success_score": 8.5, "issues": [], "reasoning": "coherent mashup"}

        config = resolve_cull_config(
            profile="ai-fun", checks=None, threshold=7.0,
            threshold_ai=None, threshold_quality=None, threshold_generation=None,
        )
        captured = io.StringIO()
        with patch.object(mod, "analyze_generation", side_effect=mock_generation), redirect_stdout(captured), redirect_stderr(io.StringIO()):
            for img in sorted(input_dir.iterdir()):
                process_image(
                    img, input_dir, "llava", config, 7.0, True, filter_dir, 512, False,
                    threading.Lock(), threading.Lock(), ScanProgress(3),
                )

        out = captured.getvalue()
        assert "Success: 8.5" in out
        assert "Success: 3.0" in out
        assert "Error processing fail.png" in out

        args = argparse.Namespace(
            model="llava",
            threshold=7.0,
            threshold_ai=None,
            threshold_quality=None,
            threshold_generation=None,
            profile="ai-fun",
            checks=None,
            dry_run=True,
            max_dimension=512,
            min_res=None,
            fast=False,
            report_path_display=None,
        )
        captured = io.StringIO()
        with patch.object(mod, "ensure_model"), patch.object(mod, "analyze_generation", side_effect=mock_generation), redirect_stdout(captured), redirect_stderr(io.StringIO()):
            run_cull(args, input_dir, filter_dir)

        meta, results = load_report(input_dir / DEFAULT_REPORT_NAME)
        assert meta["profile"] == "ai-fun"
        assert meta["max_dimension"] == 512
        by_file = {r["file"]: r for r in results}
        assert by_file["octo-rex.png"]["generation"]["success_score"] == 8.5
        assert by_file["six-finger-trex.png"]["generation"]["success_score"] == 3.0
        assert by_file["fail.png"]["status"] == "error"


def _check_hygiene_profile_cull():
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch

    from PIL import Image

    mod = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "photos"
        filter_dir = Path(tmp) / "rejects"
        input_dir.mkdir()

        payload = b"same-bytes"
        (input_dir / "keeper.jpg").write_bytes(payload)
        (input_dir / "dupe.jpg").write_bytes(payload)
        Image.new("RGB", (100, 100), "green").save(input_dir / "tiny.jpg")
        (input_dir / "bad.jpg").write_bytes(b"not-an-image")
        Image.new("RGB", (800, 600), "red").save(input_dir / "blank.png")
        good = Image.new("RGB", (800, 600))
        px = good.load()
        for y in range(600):
            for x in range(0, 800, 40):
                px[x, y] = (x + y) % 256, (128 + y) % 256, (64 + x) % 256
        good.save(input_dir / "good.jpg")

        config = resolve_cull_config(
            profile="mixed", checks=None, threshold=7.0,
            threshold_ai=None, threshold_quality=None, threshold_generation=None,
        )
        min_res = (512, 512)
        paths = [p for p in input_dir.iterdir() if p.is_file()]
        dupe_map = build_dupe_map(paths, input_dir)
        assert dupe_map == {"keeper.jpg": "dupe.jpg"}

        vlm_called: list[str] = []

        def mock_analyze(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
            vlm_called.append(img_path.name)
            return {"realism_score": 8.0, "is_realistic": True, "detected_artifacts": [], "reasoning": "ok"}

        captured = io.StringIO()
        with patch.object(mod, "analyze_image", side_effect=mock_analyze), redirect_stdout(captured), redirect_stderr(io.StringIO()):
            for img in sorted(paths):
                process_image(
                    img, input_dir, "llava", config, 7.0, True, filter_dir, 0, False,
                    threading.Lock(), threading.Lock(), ScanProgress(len(paths)),
                    dupe_of=dupe_map.get(img.relative_to(input_dir).as_posix()),
                    min_res=min_res,
                )

        out = captured.getvalue()
        assert "Hygiene: reject (dupe.jpg)" in out
        assert "below_min_res" in out
        assert "corrupt" in out
        assert "solid_color" in out
        assert vlm_called == ["good.jpg"]

        assert check_hygiene(input_dir / "bad.jpg", dupe_of=None, min_res=None)["reason"] == "corrupt"
        tiny = check_hygiene(input_dir / "tiny.jpg", dupe_of=None, min_res=(512, 512))
        assert tiny["action"] == "reject" and tiny["reason"].startswith("below_min_res")
        assert check_hygiene(input_dir / "dupe.jpg", dupe_of="keeper.jpg", min_res=None) == {
            "action": "reject", "exact_dupe_of": "keeper.jpg",
        }

        args = argparse.Namespace(
            model="llava",
            threshold=7.0,
            threshold_ai=None,
            threshold_quality=None,
            threshold_generation=None,
            profile="mixed",
            checks=("hygiene",),
            dry_run=True,
            max_dimension=0,
            min_res=(512, 512),
            fast=False,
            report_path_display=None,
        )
        captured = io.StringIO()
        with patch.object(mod, "ensure_model") as mock_ensure, redirect_stdout(captured), redirect_stderr(io.StringIO()):
            run_cull(args, input_dir, filter_dir)
        mock_ensure.assert_not_called()

        meta, results = load_report(input_dir / DEFAULT_REPORT_NAME)
        assert meta["min_res"] == "512x512"
        by_file = {r["file"]: r for r in results}
        assert by_file["keeper.jpg"]["hygiene"]["exact_dupe_of"] == "dupe.jpg"
        assert by_file["tiny.jpg"]["hygiene"]["reason"].startswith("below_min_res")
        assert by_file["bad.jpg"]["hygiene"]["reason"] == "corrupt"
        assert by_file["blank.png"]["hygiene"]["reason"] == "solid_color"


def _check_process_image_error_handling():
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch

    mod = sys.modules[__name__]
    legacy = resolve_cull_config(
        profile="mixed", checks=("ai",), threshold=7.0,
        threshold_ai=None, threshold_quality=None, threshold_generation=None,
    )
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "x.jpg"
        img.write_bytes(b"x")
        err = io.StringIO()
        out = io.StringIO()
        with patch.object(mod, "analyze_image", side_effect=RuntimeError("unexpected")), redirect_stderr(err), redirect_stdout(out):
            result = process_image(
                img, Path(tmp), "llava", legacy, 7.0, True, Path(tmp) / "rejects", 0, False,
                threading.Lock(), threading.Lock(), ScanProgress(2),
            )
        assert result == {"file": "x.jpg", "status": "error", "error": "unexpected"}
        assert "Traceback" in err.getvalue()
        assert "RuntimeError: unexpected" in err.getvalue()
        assert "[1/2] Error processing x.jpg: unexpected" in out.getvalue()


def _check_run_cull_progress_tags():
    import io
    import re
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch

    mod = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "photos"
        filter_dir = Path(tmp) / "rejects"
        input_dir.mkdir()
        for name in ("good.jpg", "bad.jpg", "fail.jpg"):
            (input_dir / name).write_bytes(b"x")

        def mock_analyze(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
            if img_path.name == "fail.jpg":
                raise RuntimeError("boom")
            score = 5.0 if img_path.name == "bad.jpg" else 8.0
            return {
                "realism_score": score,
                "is_realistic": score >= 7.0,
                "detected_artifacts": [],
                "reasoning": "test",
            }

        args = argparse.Namespace(
            model="llava",
            threshold=7.0,
            threshold_ai=None,
            threshold_quality=None,
            threshold_generation=None,
            profile="mixed",
            checks=("ai",),
            dry_run=False,
            max_dimension=0,
            min_res=None,
            fast=False,
            report_path_display=None,
        )
        captured = io.StringIO()
        with patch.object(mod, "ensure_model"), patch.object(mod, "analyze_image", side_effect=mock_analyze), redirect_stdout(captured), redirect_stderr(io.StringIO()):
            run_cull(args, input_dir, filter_dir)

        out = captured.getvalue()
        analyzing_tags = re.findall(r"(\[\d+/3\]) Analyzing (\S+?)\.\.\.", out)
        assert len(analyzing_tags) == 3
        assert len({tag for tag, _ in analyzing_tags}) == 3

        by_tag = {tag: fname for tag, fname in analyzing_tags}
        for tag, fname in by_tag.items():
            tag_lines = [line for line in out.splitlines() if line.startswith(tag)]
            if fname == "fail.jpg":
                assert any("Error processing fail.jpg" in line for line in tag_lines)
            else:
                assert any("Score:" in line for line in tag_lines)
            if fname == "bad.jpg":
                assert any("Moved filtered image" in line for line in tag_lines)
            elif fname == "good.jpg":
                assert any("Preserved keeper" in line for line in tag_lines)

        _, results = load_report(input_dir / DEFAULT_REPORT_NAME)
        assert len(results) == 3
        assert sum(entry.get("status") == "error" for entry in results) == 1


def main():
    args = parse_args()
    if args.self_check:
        _self_check()
        return

    input_dir = Path(args.dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    filter_dir = Path(args.filter_dir) if args.filter_dir else input_dir / "rejects"

    if args.apply_report is not None:
        explicit = None if args.apply_report == "__default__" else Path(args.apply_report)
        report_path = resolve_report_path(input_dir, explicit)
        if not report_path.is_file():
            names = ", ".join(LEGACY_REPORT_NAMES)
            raise SystemExit(
                f"Error: Report file '{report_path}' does not exist "
                f"(also checked legacy names: {names})."
            )
        apply_from_report(
            report_path,
            input_dir,
            filter_dir,
            threshold=args.threshold,
            threshold_ai=args.threshold_ai,
            threshold_quality=args.threshold_quality,
            threshold_generation=args.threshold_generation,
            force_reapply=args.force_reapply,
        )
        return

    run_cull(args, input_dir, filter_dir)



def _check_edit_policy():
    assert get_original_stem("IMG_1234-edited") == "IMG_1234"
    assert get_original_stem("PXL_1234_edited") == "PXL_1234"
    assert get_original_stem("IMG_E1234") == "IMG_1234"
    assert get_original_stem("IMG_E1234-edited") == "IMG_1234"
    assert get_original_stem("normal") is None

    p1 = Path("IMG_1234.jpg")
    p2 = Path("IMG_1234-edited.jpg")
    p3 = Path("IMG_E1234.HEIC")
    p4 = Path("IMG_1234.HEIC")
    p5 = Path("IMG_E1234-edited.jpg")
    paths = [p1, p2, p3, p4, p5]
    
    orig_map = build_edit_policy_map(paths, Path("."), "original")
    assert orig_map[p2.as_posix()] == "edit_policy (prefer original)"
    assert orig_map[p3.as_posix()] == "edit_policy (prefer original)"
    assert orig_map[p5.as_posix()] == "edit_policy (prefer original)"
    assert p1.as_posix() not in orig_map
    assert p4.as_posix() not in orig_map

    edited_map = build_edit_policy_map(paths, Path("."), "edited")
    assert edited_map[p1.as_posix()] == "edit_policy (prefer edited)"
    assert edited_map[p4.as_posix()] == "edit_policy (prefer edited)"
    assert p2.as_posix() not in edited_map
    assert p3.as_posix() not in edited_map
    assert p5.as_posix() not in edited_map


def _check_recursive_cull():
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch
    mod = sys.modules[__name__]

    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        input_dir.mkdir()
        sub = input_dir / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "nested.jpg").write_bytes(b"x")
        (input_dir / "top.jpg").write_bytes(b"x")
        
        filter_dir = input_dir / "rejects"
        filter_sub = filter_dir / "sub"
        filter_sub.mkdir(parents=True)
        (filter_sub / "rejected.jpg").write_bytes(b"x")

        # mock so we can just trace files without ollama logic breaking
        def mock_process(img_path, *args, **kwargs):
            return {"file": img_path.relative_to(input_dir).as_posix()}
        
        args = argparse.Namespace(
            dir=str(input_dir),
            filter_dir=str(filter_dir),
            recursive=True,
            model="llava",
            threshold=5.0,
            threshold_ai=None,
            threshold_quality=None,
            threshold_generation=None,
            profile=None,
            checks=None,
            dry_run=True,
            min_res=None,
            max_dimension=0,
            fast=False,
            self_check=False,
            apply_report=None,
            report_path_display=None,
            force_reapply=False
        )

        with (
            patch.object(mod, "ensure_model"),
            patch.object(mod, "process_image", side_effect=mock_process),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO())
        ):
            run_cull(args, input_dir, filter_dir)
        
        report_path = input_dir / DEFAULT_REPORT_NAME
        _, results = load_report(report_path)
        files = [r["file"] for r in results]
        
        assert len(files) == 2, f"Expected 2 files, got {len(files)}"
        assert "top.jpg" in files
        assert "sub/deep/nested.jpg" in files
        assert "rejects/sub/rejected.jpg" not in files, "Filter dir should be excluded"

if __name__ == "__main__":
    main()
