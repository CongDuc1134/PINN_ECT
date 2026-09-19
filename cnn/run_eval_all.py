import os
import sys
import traceback
import faulthandler
import subprocess
import re
import time
import threading
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
import math
from collections import defaultdict
import glob
import yaml
import tempfile
import hashlib
import inspect

try:
    faulthandler.enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Force tempdir to /work/dat.lt19010205/Cong_Duc/check/tmp or local /tmp to prevent NFS '.nfsXXX' file locking
_work_tmp = '/work/dat.lt19010205/Cong_Duc/check/tmp'
for _tmp_candidate in [_work_tmp, '/tmp', '/var/tmp']:
    try:
        if _tmp_candidate == _work_tmp and not os.path.exists(_tmp_candidate):
            os.makedirs(_tmp_candidate, exist_ok=True)
        if os.path.exists(_tmp_candidate) and os.access(_tmp_candidate, os.W_OK):
            os.environ['TMPDIR'] = _tmp_candidate
            tempfile.tempdir = _tmp_candidate
            break
    except Exception:
        pass

# Patch multiprocessing.util._remove_temp_dir to safely ignore NFS Errno 16 errors
try:
    import multiprocessing.util as _mp_util
    import shutil as _shutil

    def _safe_remove_temp_dir(tempdir):
        try:
            _shutil.rmtree(tempdir, ignore_errors=True)
        except Exception:
            pass

    _mp_util._remove_temp_dir = _safe_remove_temp_dir
except Exception:
    pass


def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    script_name = os.path.basename(__file__) if '__file__' in globals() else 'run_eval_all.py'
    print(f"\n[FATAL] Unhandled exception in {script_name}", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb)


sys.excepthook = _log_uncaught_exception

# Import experiment manager
try:
    from experiment_manager import ExperimentManager, ReportGenerator
except ImportError:
    ExperimentManager = None
    ReportGenerator = None

# --- CONFIGURATION ---
def _get_int_env(name, default, min_value=0):
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got: {value}")
    return value


def _get_bool_env(name, default=True):
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like (0/1/true/false), got: {raw}")


def _parse_alpha_values(raw):
    values = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            val = float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid alpha value: {text}") from exc
        if val < 0:
            raise ValueError(f"Alpha must be >= 0, got: {val}")
        values.append(val)
    if not values:
        raise ValueError("ALPHA_VALUES is empty")
    return values


def _parse_group_values(raw):
    groups = []
    allowed = {"no_pinn", "baseline", "alpha", "warmup", "seed"}
    for chunk in raw.split(","):
        text = chunk.strip().lower()
        if not text:
            continue
        if text not in allowed:
            raise ValueError(f"Invalid TEST_GROUPS entry: {text}")
        groups.append(text)
    if not groups:
        raise ValueError("TEST_GROUPS is empty")
    return groups


def _parse_int_list(raw, name, min_value=None, max_value=None):
    values = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            val = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} entry: {text}") from exc
        if min_value is not None and val < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got: {val}")
        if max_value is not None and val > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got: {val}")
        values.append(val)
    if not values:
        raise ValueError(f"{name} is empty")
    return values


def _parse_seed_list(raw):
    seeds = [s.strip() for s in raw.split(",") if s.strip()]
    if not seeds:
        raise ValueError("Seed list is empty")
    return seeds


def _parse_csv_strings(raw, default):
    values = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
    return values or list(default)


def _atomic_to_csv(df, path, **kwargs):
    """Write DataFrame to CSV atomically (write to temp then replace).

    Ensures consumers never see a partially written file.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dirname)
    os.close(fd)
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _atomic_to_text(text, path, encoding="utf-8"):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dirname)
    os.close(fd)
    try:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _file_md5(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


PLOT_METRICS_ONLY = "--plot-metrics-only" in sys.argv
if PLOT_METRICS_ONLY:
    sys.argv.remove("--plot-metrics-only")

EVAL_PRESET = os.environ.get("EVAL_PRESET", "all_existing").strip().lower()
if EVAL_PRESET in {"default", "full"}:
    EVAL_PRESET = ""


if EVAL_PRESET == "all_existing":
    # Default command: infer/refresh every main case already trained, then
    # train only configurations whose checkpoints are still missing.
    os.environ["TEST_GROUPS"] = "baseline,alpha,seed"
    os.environ.setdefault("REPORT_TAG", "comparison")
    os.environ["PERCENT_LIST"] = "1,3,5,7,10"
    os.environ["AUTO_SELECT_BEST_ALPHA"] = "0"
    os.environ["BEST_ALPHA"] = "1"
    os.environ["ALPHA_TEST_VALUES"] = "1,10,100,1000"
    os.environ["ALPHA_TEST_WARMUP"] = "100"
    os.environ["ALPHA_TEST_PERCENTS"] = "5"
    os.environ["ALPHA_TEST_SEEDS"] = "42"
    os.environ["SEED_TEST_ALPHA"] = "1"
    os.environ["SEED_TEST_WARMUP"] = "100"
    os.environ["SEED_TEST_PERCENTS"] = "5"
    os.environ["SEED_TEST_SEEDS"] = "42,123,456,789"
    os.environ["INFER_BEFORE_TRAIN"] = "0"




DEFAULT_PERCENT_LIST = "1,3,5,7,10"
DEFAULT_SEED_TEST_PERCENT = "5"

START_PERCENT = _get_int_env("START_PERCENT", 1, min_value=1)
END_PERCENT = _get_int_env("END_PERCENT", 10, min_value=1)
BATCH_SIZE = _get_int_env("BATCH_SIZE", 8, min_value=1)
EPOCHS = _get_int_env("EPOCHS", 300, min_value=1)
# Default to two concurrent training jobs on the same GPU for faster sweeps.
# Override with PARALLEL_RUNS/GPU_IDS env vars when needed.
PARALLEL_RUNS = _get_int_env("PARALLEL_RUNS", 1, min_value=1)
PINN_ACTIVATION_EPOCH = _get_int_env("PINN_ACTIVATION_EPOCH", 100, min_value=0)
SEEDS = [s.strip() for s in os.environ.get("SEEDS", "42,123,456").split(',') if s.strip()]
if not SEEDS:
    SEEDS = ["42"]
PINN_LOSS_TYPE = os.environ.get("PINN_LOSS_TYPE", "log1p_norm_sse").strip()
RUN_TRAIN = _get_bool_env("RUN_TRAIN", True)
# Reuse completed models by default. Completed runs are skipped during the
# training phase, then loaded again by the re-evaluation phase for inference,
# metric refresh, and plotting. Incomplete runs resume from their latest
# checkpoint instead of starting over.
SKIP_IF_EXISTS = _get_bool_env("SKIP_IF_EXISTS", True)
AUTO_RESUME = _get_bool_env("AUTO_RESUME", True)
REUSE_COMPLETED_NO_PINN = _get_bool_env("REUSE_COMPLETED_NO_PINN", True)
LIVE_PROGRESS_PLOTS = _get_bool_env("LIVE_PROGRESS_PLOTS", True)
INFER_BEFORE_TRAIN = _get_bool_env("INFER_BEFORE_TRAIN", False)
REEVAL_IF_MODEL_EXISTS = _get_bool_env("REEVAL_IF_MODEL_EXISTS", True)
REEVAL_WRITE_PREDICTIONS = _get_bool_env("REEVAL_WRITE_PREDICTIONS", True)
REPORT_REEVAL_ONLY = _get_bool_env("REPORT_REEVAL_ONLY", (not RUN_TRAIN and REEVAL_IF_MODEL_EXISTS))
FORCE_REEVAL = _get_bool_env("FORCE_REEVAL", False)
GPU_IDS = _parse_csv_strings(
    os.environ.get("GPU_IDS", "0"),
    default=["0"],
)
# Alpha values included in the default PINN sweep.
# Override with ALPHA_VALUES when a custom sweep is needed.
ALPHA_VALUES = _parse_alpha_values(
    os.environ.get("ALPHA_VALUES", "1,10,100,1000")
)
# Remove alpha==0 since that's equivalent to No-PINN
if 0 in ALPHA_VALUES:
    ALPHA_VALUES = [a for a in ALPHA_VALUES if a != 0]


if START_PERCENT > END_PERCENT:
    raise ValueError(f"Invalid range: START_PERCENT={START_PERCENT} > END_PERCENT={END_PERCENT}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

REP_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "rep_code"))
if REP_CODE_DIR not in sys.path:
    sys.path.insert(0, REP_CODE_DIR)

os.chdir(SCRIPT_DIR)
PYTHON_CMD = sys.executable  # Use current conda python
SCRIPT_NO_PINN = os.path.join(SCRIPT_DIR, "main_percent_no_pinn.py")
SCRIPT_PINN = os.path.join(SCRIPT_DIR, "main_percent_new.py")

if EVAL_PRESET == "all_existing":
    default_study_name = "all_existing_main_cases"

else:
    enabled_groups = os.environ.get("TEST_GROUPS", "no_pinn,alpha,seed")
    default_study_name = "custom_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", enabled_groups).strip("_")

STUDY_NAME = os.environ.get("STUDY_NAME", default_study_name).strip()
if not STUDY_NAME:
    raise ValueError("STUDY_NAME must not be empty")
OUTPUT_ROOT = os.path.abspath(
    os.environ.get("EVAL_OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "Outputs_cnn_eval"))
)
STUDY_ROOT = os.path.join(OUTPUT_ROOT, STUDY_NAME)
PINN_MODEL_ROOT = os.path.join(SCRIPT_DIR, "Outputs_cnn_pinn")
NO_PINN_MODEL_ROOT = os.path.join(SCRIPT_DIR, "Outputs_cnn_baseline")
OUTPUT_PINN_ROOT = os.path.join(PINN_MODEL_ROOT, f"loss_{PINN_LOSS_TYPE}")
OUTPUT_NO_PINN_ROOT = NO_PINN_MODEL_ROOT
REPORT_ROOT = os.path.join(STUDY_ROOT, "reports")
os.environ["PINN_OUTPUT_ROOT"] = PINN_MODEL_ROOT
os.environ["NO_PINN_OUTPUT_ROOT"] = NO_PINN_MODEL_ROOT

CHECK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.abspath(os.path.join(CHECK_DIR, ".."))

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH or not os.path.exists(DATA_PATH):
    for candidate in [
        os.path.join(CHECK_DIR, "Crack_Shape_Images"),
        os.path.join(SCRIPT_DIR, "Crack_Shape_Images"),
        os.path.join(REP_CODE_DIR, "Crack_Shape_Images"),
        os.path.join(PAPER_DIR, "Crack_Shape_Images"),
    ]:
        if os.path.exists(candidate):
            DATA_PATH = candidate
            break

LABELS_PATH = os.environ.get("LABELS_PATH")
if not LABELS_PATH or not os.path.exists(LABELS_PATH):
    for candidate in [
        os.path.join(CHECK_DIR, "labels.csv"),
        os.path.join(SCRIPT_DIR, "labels.csv"),
        os.path.join(REP_CODE_DIR, "labels.csv"),
        os.path.join(PAPER_DIR, "labels.csv"),
    ]:
        if os.path.exists(candidate):
            LABELS_PATH = candidate
            break

if DATA_PATH and os.path.exists(DATA_PATH):
    os.environ["DATA_PATH"] = DATA_PATH
if LABELS_PATH and os.path.exists(LABELS_PATH):
    os.environ["LABELS_PATH"] = LABELS_PATH
DEFAULT_REPORT_TAG = "run_20260610_102220"
REPORT_TAG = os.environ.get("REPORT_TAG", DEFAULT_REPORT_TAG).strip()
REPORT_DIR = os.path.abspath(
    os.environ.get("REPORT_DIR", os.path.join(REPORT_ROOT, REPORT_TAG))
)
BY_PERCENT_DIR = os.path.join(REPORT_DIR, "by_percent")
os.makedirs(BY_PERCENT_DIR, exist_ok=True)
RUN_UNITS_DONE = set()
RUN_UNITS_LOCK = threading.Lock()

# Percents to include in final reports. Keep this aligned with the group defaults below.
PERCENT_LIST = [int(p.strip()) for p in os.environ.get("PERCENT_LIST", DEFAULT_PERCENT_LIST).split(",") if p.strip()]

# Initialize experiment manager and report generator
EXP_MANAGER = None
REPORT_GEN = None
if ExperimentManager and ReportGenerator:
    EXP_MANAGER = ExperimentManager(SCRIPT_DIR, shortname="ablation_alpha_warmup")
    REPORT_GEN = ReportGenerator(os.path.join(SCRIPT_DIR, "Result"))


def alpha_to_label(alpha):
    # Keep enough precision to avoid collisions like 1.5e-2 vs 2e-2.
    # Using general format also keeps labels readable in plots/logs.
    return f"{alpha:.12g}"


def alpha_to_tag(alpha):
    # filesystem-friendly tag for run names
    return (
        alpha_to_label(alpha)
        .replace(".", "p")
        .replace("+", "p")
        .replace("-", "m")
    )


def make_series_item(
    *,
    key,
    label,
    short_label,
    mode,
    script,
    tag,
    env,
    color,
    linestyle,
    marker,
    percent_list=None,
    seed_list=None,
    alpha=None,
    warmup=None,
    group=None,
):
    item = {
        "key": key,
        "label": label,
        "short_label": short_label,
        "mode": mode,
        "script": script,
        "tag": tag,
        "env": env,
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "percent_list": list(percent_list) if percent_list is not None else list(PERCENT_LIST),
        "seed_list": [str(s) for s in seed_list] if seed_list is not None else list(SEEDS),
        "group": group or "main",
    }
    if alpha is not None:
        item["alpha"] = alpha
    if warmup is not None:
        item["warmup"] = warmup
    return item


DISTINCT_SERIES_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#393b79",  # indigo
    "#637939",  # dark olive
    "#8c6d31",  # ochre
    "#843c39",  # brick
    "#7b4173",  # plum
    "#3182bd",  # steel blue
    "#31a354",  # forest green
    "#e6550d",  # rust
    "#756bb1",  # lavender
    "#636363",  # charcoal
]

DISTINCT_SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]


def scan_existing_series():
    """Dynamically scan output directories to build the series list from actual existing runs."""
    import re
    
    def tag_to_alpha(a_tag):
        try:
            return float(a_tag.replace('p', '.'))
        except ValueError:
            return None

    def parse_tag(tag):
        if tag.startswith("NoPINN_seed_"):
            return {"mode": "no_pinn", "group": "seed"}
        elif tag.startswith("NoPINN"):
            return {"mode": "no_pinn", "group": "baseline"}
        elif tag.startswith("PINN_base_a"):
            match = re.match(r"PINN_base_a([0-9p]+)_W(\d+)_E\d+", tag)
            if match:
                return {"mode": "pinn", "group": "baseline", "alpha": tag_to_alpha(match.group(1)), "warmup": int(match.group(2))}
        elif tag.startswith("PINN_alpha_a"):
            match = re.match(r"PINN_alpha_a([0-9p]+)_W(\d+)_E\d+", tag)
            if match:
                return {"mode": "pinn", "group": "alpha", "alpha": tag_to_alpha(match.group(1)), "warmup": int(match.group(2))}
        elif tag.startswith("PINN_warmup_a"):
            match = re.match(r"PINN_warmup_a([0-9p]+)_W(\d+)_E\d+", tag)
            if match:
                return {"mode": "pinn", "group": "warmup", "alpha": tag_to_alpha(match.group(1)), "warmup": int(match.group(2))}
        elif tag.startswith("PINN_seed_a"):
            match = re.match(r"PINN_seed_a([0-9p]+)_W(\d+)_E\d+", tag)
            if match:
                return {"mode": "pinn", "group": "seed", "alpha": tag_to_alpha(match.group(1)), "warmup": int(match.group(2))}
        return None

    tags_info = {}

    def scan_dir(root_dir):
        if not os.path.isdir(root_dir):
            return
        for pct_folder in os.listdir(root_dir):
            match = re.match(r"train_(\d+)pct", pct_folder)
            if not match:
                continue
            pct = int(match.group(1))
            
            pct_path = os.path.join(root_dir, pct_folder)
            if not os.path.isdir(pct_path):
                continue
            for run_folder in os.listdir(pct_path):
                match_run = re.match(r"^(.*)_seed_([0-9a-zA-Z]+)(?:_run_.*)?$", run_folder)
                if match_run:
                    tag = match_run.group(1)
                    seed = match_run.group(2)
                    
                    if tag not in tags_info:
                        parsed = parse_tag(tag)
                        if parsed is None:
                            continue
                        tags_info[tag] = {
                            "parsed": parsed,
                            "seeds": set(),
                            "percents": set()
                        }
                    tags_info[tag]["seeds"].add(seed)
                    tags_info[tag]["percents"].add(pct)

    scan_dir(OUTPUT_NO_PINN_ROOT)
    scan_dir(OUTPUT_PINN_ROOT)

    new_series = []
    
    for tag in sorted(tags_info.keys()):
        info = tags_info[tag]
        parsed = info["parsed"]
        mode = parsed["mode"]
        group = parsed["group"]
        
        if mode == "no_pinn":
            new_series.append(
                make_series_item(
                    key=tag,
                    label=f"No PINN baseline (E={EPOCHS})" if group == "baseline" else f"No PINN {group}",
                    short_label="No PINN",
                    mode="no_pinn",
                    script=SCRIPT_NO_PINN,
                    tag=tag,
                    env={"_MODE": "no_pinn", "EPOCHS": str(EPOCHS), "SERIES_LABEL": f"No PINN {group}"},
                    color="#1f77b4", 
                    linestyle="--",
                    marker="o",
                    percent_list=sorted(info["percents"]),
                    seed_list=sorted(info["seeds"]),
                    group=group,
                )
            )
        else:
            alpha = parsed.get("alpha")
            warmup = parsed.get("warmup")
            a_label = alpha_to_label(alpha) if alpha is not None else "None"
            
            if group == "baseline":
                label = f"PINN baseline a={a_label} W={warmup}"
            elif group == "warmup":
                label = f"PINN a={a_label} W={warmup}"
            else:
                label = f"PINN alpha={a_label} W={warmup}"
                
            short_label = f"a={a_label}" if group == "alpha" else (f"W={warmup}" if group == "warmup" else f"PINN\na={a_label} W={warmup}")
            
            new_series.append(
                make_series_item(
                    key=tag,
                    label=label,
                    short_label=short_label,
                    mode="pinn",
                    script=SCRIPT_PINN,
                    tag=tag,
                    env={
                        "_MODE": "pinn",
                        "EPOCHS": str(EPOCHS),
                        "PINN_ACTIVATION_EPOCH": str(warmup),
                        "ALPHA_INIT": f"{alpha:.12g}" if alpha is not None else "",
                        "SERIES_LABEL": label,
                    },
                    color="#d62728",
                    linestyle="-",
                    marker="s",
                    percent_list=sorted(info["percents"]),
                    seed_list=sorted(info["seeds"]),
                    alpha=alpha,
                    warmup=warmup,
                    group=group,
                )
            )
            
    return new_series

def apply_distinct_series_styles(series_items):
    """Assign globally distinct plot styles so legends remain readable."""
    for idx, item in enumerate(series_items):
        item["color"] = DISTINCT_SERIES_COLORS[idx % len(DISTINCT_SERIES_COLORS)]
        item["marker"] = DISTINCT_SERIES_MARKERS[idx % len(DISTINCT_SERIES_MARKERS)]
        if item.get("mode") == "no_pinn":
            item["linestyle"] = "--"
        elif item.get("group") == "alpha":
            item["linestyle"] = "-"
        elif item.get("group") == "warmup":
            item["linestyle"] = "-."
        elif item.get("group") == "seed":
            item["linestyle"] = ":"
        else:
            item["linestyle"] = "-"


def collapse_equivalent_plot_series(series_items):
    """Collapse aliases that intentionally reuse the same training output."""
    collapsed = []
    seen = {}
    for item in series_items:
        sig = (
            item.get("mode"),
            item.get("tag"),
            tuple(str(s) for s in item.get("seed_list", [])),
        )
        if sig not in seen:
            plot_item = dict(item)
            plot_item["_plot_alias_labels"] = [item["label"]]
            seen[sig] = plot_item
            collapsed.append(plot_item)
            continue

        existing = seen[sig]
        existing["_plot_alias_labels"].append(item["label"])
    return collapsed


def get_percent_dir(mode, percent):
    pct_dir = f"train_{percent:02d}pct"
    if mode == "no_pinn":
        return os.path.join(OUTPUT_NO_PINN_ROOT, pct_dir)
    return os.path.join(OUTPUT_PINN_ROOT, pct_dir)


def completed_checkpoint_candidates(run_dir):
    """Return model files that prove a run has a valid trained checkpoint."""
    return [
        os.path.join(run_dir, "checkpoint_best_post_pinn", "best_model_pytorch.pth"),
        os.path.join(run_dir, "best_model_pytorch_post_pinn.pth"),
        os.path.join(run_dir, "checkpoint_best", "best_model_pytorch.pth"),
        os.path.join(run_dir, "best_model_pytorch.pth"),
        os.path.join(run_dir, "checkpoint_final", "checkpoint_full.pth"),
        os.path.join(run_dir, "checkpoint_final", "model_final.pth"),
        os.path.join(run_dir, "checkpoint_latest", "checkpoint_full.pth"),
        os.path.join(run_dir, "checkpoint_latest", "model_latest.pth"),
    ]


def has_completed_checkpoint(run_dir):
    return any(os.path.isfile(path) for path in completed_checkpoint_candidates(run_dir))


def find_latest_run_dir(percent_dir, tag_prefix):
    if not os.path.isdir(percent_dir):
        return None
    candidates = [
        name for name in os.listdir(percent_dir)
        if os.path.isdir(os.path.join(percent_dir, name)) and name.startswith(tag_prefix + "_")
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    # Prefer a completed model even when its CSV is missing. Re-evaluation will
    # recreate all metric files from this checkpoint.
    for name in candidates:
        run_dir = os.path.join(percent_dir, name)
        if has_completed_checkpoint(run_dir):
            return run_dir
    for name in candidates:
        run_dir = os.path.join(percent_dir, name)
        csv_path = os.path.join(run_dir, "training_results.csv")
        if os.path.isfile(csv_path):
            return run_dir
    return os.path.join(percent_dir, candidates[0])


def has_completed_result(mode, percent, tag_prefix):
    percent_dir = get_percent_dir(mode, percent)
    run_dir = find_latest_run_dir(percent_dir, tag_prefix)
    if run_dir is None:
        return False
    return has_completed_checkpoint(run_dir)


def has_completed_seed_result(mode, percent, seed_tag):
    """Check whether a specific seed run for a percent has already completed."""
    return has_completed_result(mode, percent, seed_tag)


def metric_csv_run_dir(csv_path):
    """Return the run directory that owns a metric CSV."""
    norm = os.path.normpath(csv_path)
    parts = norm.split(os.sep)
    if "checkpoint_final" in parts:
        idx = parts.index("checkpoint_final")
        prefix = os.sep if norm.startswith(os.sep) else ""
        return os.path.normpath(prefix + os.path.join(*parts[:idx]))
    return os.path.normpath(os.path.dirname(norm))


def is_report_csv_allowed(csv_path):
    if not REPORT_REEVAL_ONLY:
        return True
    return metric_csv_run_dir(csv_path) in _REEVAL_SUCCESS_RUN_DIRS


def filter_preferred_metric_csvs(files):
    """Prefer run-root metric CSVs over identical copies in checkpoint_final."""
    normalized = [os.path.normpath(p) for p in files]
    available = set(normalized)
    selected = []
    for path in sorted(normalized):
        if not is_report_csv_allowed(path):
            continue
        parts = path.split(os.sep)
        if "checkpoint_final" in parts:
            idx = parts.index("checkpoint_final")
            root_copy = os.path.join(*parts[:idx], *parts[idx + 1:])
            if path.startswith(os.sep):
                root_copy = os.sep + root_copy
            if root_copy in available:
                continue
        selected.append(path)
    return selected


def find_latest_checkpoint_for_percent(mode, percent, tag_prefix=None):
    """Find a checkpoint file to resume from for given mode and percent.

    Returns:
      - "__COMPLETED__" if a final checkpoint is present (indicating completed run)
      - path to latest epoch checkpoint (checkpoint_full.pth) if found
      - None if nothing found
    """
    if mode == "no_pinn":
        folder_root = OUTPUT_NO_PINN_ROOT
    else:
        folder_root = OUTPUT_PINN_ROOT

    pct_dir = os.path.join(folder_root, f"train_{percent:02d}pct")
    if not os.path.isdir(pct_dir):
        return None

    # Prefer any completed model over a newer partial duplicate.
    run_dirs = sorted(
        [
            d for d in os.listdir(pct_dir)
            if os.path.isdir(os.path.join(pct_dir, d))
            and (not tag_prefix or tag_prefix in d)
        ],
        reverse=True,
    )
    for run_name in run_dirs:
        run_dir = os.path.join(pct_dir, run_name)
        final_ckpts = [
            path for path in completed_checkpoint_candidates(run_dir)
            if os.path.isfile(path)
        ]
        if final_ckpts:
            if SKIP_IF_EXISTS:
                return "__COMPLETED__"
            return final_ckpts[0]

    # No completed model exists, so resume the newest partial checkpoint.
    for run_name in run_dirs:
        run_dir = os.path.join(pct_dir, run_name)
        # find latest epoch checkpoint inside this run_dir
        pattern = os.path.join(run_dir, 'checkpoint_epoch_*', 'checkpoint_full.pth')
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]

    return None


def _series_train_signature(item):
    """Return the actual train configuration, excluding labels/report grouping."""
    env = item.get("env", {})
    mode = item.get("mode", env.get("_MODE", "pinn"))
    epochs = int(env.get("EPOCHS", EPOCHS))
    script = item.get("script", "")
    if mode == "no_pinn":
        return (mode, script, epochs)

    alpha = float(env.get("ALPHA_INIT", item.get("alpha", 0.0)))
    warmup = int(env.get("PINN_ACTIVATION_EPOCH", item.get("warmup", PINN_ACTIVATION_EPOCH)))
    return (mode, script, epochs, PINN_LOSS_TYPE, alpha_to_label(alpha), warmup)


def dedupe_equivalent_series_tags(series_items):
    """Make equivalent train configs share one tag so they are not trained twice."""
    seen = {}
    for item in series_items:
        sig = _series_train_signature(item)
        if sig not in seen:
            seen[sig] = item
            continue

        canonical = seen[sig]
        old_tag = item["tag"]
        item["tag"] = canonical["tag"]
        item["_deduped_from"] = canonical["key"]
        print(
            f"[DEDUP] {item['key']} shares train config with {canonical['key']}; "
            f"reuse tag '{canonical['tag']}' instead of '{old_tag}'"
        )


def collect_series_results(series_items):
    """Populate per-seed and aggregated results for each series item."""
    all_percents = set()
    for item in series_items:
        per_seed_results = {}
        seed_list = item.get('seed_list', SEEDS)
        allowed_percents = {
            int(p) for p in item.get("percent_list", PERCENT_LIST)
        }
        for seed in seed_list:
            seed_suffix = f"{item['tag']}_seed_{seed}"
            res = {
                pct: rec
                for pct, rec in extract_metrics(item["mode"], seed_suffix).items()
                if int(pct) in allowed_percents
            }
            per_seed_results[seed] = res
            all_percents.update(res.keys())
        item["per_seed_results"] = per_seed_results
        item["results"] = aggregate_seed_results(per_seed_results)
    return all_percents


def choose_best_alpha_from_alpha_series(series_items, percent_priority):
    """Choose the alpha with lowest mean NRMSE/RMSE over the same percent set."""
    target_pcts = sorted({int(p) for p in percent_priority})
    candidates = []
    excluded = []
    for item in series_items:
        if item.get("group") != "alpha" or item.get("mode") != "pinn":
            continue
        alpha, _ = get_series_alpha_warmup(item)
        if alpha is None:
            continue
        results = item.get("results", {})
        if not results:
            excluded.append((float(alpha), item["key"], target_pcts or [], "no completed results"))
            continue

        pcts = target_pcts or sorted(int(p) for p in results.keys())
        vals = []
        missing = []
        for pct in pcts:
            try:
                res = results[pct]
                val = float(res.get("nrmse", res.get("nrmse_range_avg", res.get("rmse", np.nan))))
                if np.isfinite(val):
                    vals.append(val)
                else:
                    missing.append(pct)
            except Exception:
                missing.append(pct)

        if missing or len(vals) != len(pcts):
            excluded.append((float(alpha), item["key"], missing, "missing NRMSE/RMSE"))
            continue

        if vals and pcts:
            candidates.append((float(np.mean(vals)), float(alpha), item["key"], pcts))

    if not candidates:
        if excluded:
            print("[AUTO][WARN] No alpha has complete NRMSE coverage for best-alpha selection.")
            for alpha, key, missing, reason in excluded[:10]:
                print(f"[AUTO][WARN] Exclude alpha={alpha_to_label(alpha)} ({key}): {reason}, percents={missing}")
        return None

    for alpha, key, missing, reason in excluded:
        print(f"[AUTO][INFO] Exclude alpha={alpha_to_label(alpha)} ({key}): {reason}, percents={missing}")

    candidates.sort(key=lambda x: (x[0], x[1]))
    best_score, best_alpha, best_key, best_pcts = candidates[0]
    print(
        f"[AUTO] Best alpha from alpha sweep: alpha={alpha_to_label(best_alpha)} "
        f"(mean NRMSE={best_score:.6g}%, source={best_key}, percents={best_pcts})"
    )
    return best_alpha


def retarget_deferred_pinn_alpha(series_items, selected_alpha, *, update_warmup, update_seed):
    """Retarget warmup/seed PINN studies to the selected alpha unless env overrode them."""
    a_label = alpha_to_label(selected_alpha)
    a_tag = alpha_to_tag(selected_alpha)
    for item in series_items:
        if item.get("mode") != "pinn":
            continue
        group = item.get("group")
        if group == "warmup" and update_warmup:
            warmup = int(item["env"]["PINN_ACTIVATION_EPOCH"])
            item["alpha"] = selected_alpha
            item["label"] = f"PINN a={a_label} W={warmup}"
            item["short_label"] = f"W={warmup}"
            item["tag"] = f"PINN_warmup_a{a_tag}_W{warmup}_E{EPOCHS}"
            item["env"]["ALPHA_INIT"] = f"{selected_alpha:.12g}"
            item["env"]["SERIES_LABEL"] = f"PINN alpha={a_label} W={warmup}"
        elif group == "seed" and update_seed:
            warmup = int(item["env"]["PINN_ACTIVATION_EPOCH"])
            seed_label = ",".join(str(s) for s in item.get("seed_list", [])) or "unknown"
            item["alpha"] = selected_alpha
            item["label"] = f"PINN seed={seed_label} a={a_label} W={warmup}"
            item["short_label"] = f"PINN\nseed={seed_label}"
            item["tag"] = f"PINN_seed_a{a_tag}_W{warmup}_E{EPOCHS}"
            item["env"]["ALPHA_INIT"] = f"{selected_alpha:.12g}"
            item["env"]["SERIES_LABEL"] = f"PINN seed={seed_label} a={a_label} W={warmup}"


def get_series_alpha_warmup(item):
    """Return numeric (alpha, warmup) metadata for PINN heatmap plots."""
    if item.get("mode") != "pinn":
        return None, None

    env = item.get("env", {})
    alpha = item.get("alpha")
    warmup = item.get("warmup")

    if alpha is None and "ALPHA_INIT" in env:
        try:
            alpha = float(env["ALPHA_INIT"])
        except Exception:
            alpha = None

    if warmup is None and "PINN_ACTIVATION_EPOCH" in env:
        try:
            warmup = int(env["PINN_ACTIVATION_EPOCH"])
        except Exception:
            warmup = None

    return alpha, warmup


_REEVAL_BASE_DATA = None
_REEVAL_SPLIT_CACHE = {}
_REEVAL_SUCCESS_RUN_DIRS = set()


def _reeval_imports():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        explained_variance_score,
        f1_score,
        max_error,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        matthews_corrcoef,
        precision_recall_fscore_support,
        precision_score,
        r2_score,
        recall_score,
    )
    from library_functions import Load_Data_With_Labels
    return {
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
        "train_test_split": train_test_split,
        "StandardScaler": StandardScaler,
        "balanced_accuracy_score": balanced_accuracy_score,
        "cohen_kappa_score": cohen_kappa_score,
        "confusion_matrix": confusion_matrix,
        "explained_variance_score": explained_variance_score,
        "f1_score": f1_score,
        "max_error": max_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "median_absolute_error": median_absolute_error,
        "matthews_corrcoef": matthews_corrcoef,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "precision_score": precision_score,
        "r2_score": r2_score,
        "recall_score": recall_score,
        "Load_Data_With_Labels": Load_Data_With_Labels,
    }


def _build_reeval_model_class(torch, nn):
    class ImprovedMultimodelNet(nn.Module):
        def __init__(self, num_shapes):
            super().__init__()
            # Learnable uncertainty parameters
            self.log_var_clf = nn.Parameter(torch.tensor(0.0))
            self.log_var_w = nn.Parameter(torch.tensor(0.0))
            self.log_var_l = nn.Parameter(torch.tensor(0.0))
            self.log_var_d = nn.Parameter(torch.tensor(0.0))

            self.backbone = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.1),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Dropout(0.1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_shapes),
            )
            self.regressor_backbone = nn.Sequential(
                nn.Linear(128 * 4 * 4, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.1),
            )
            self.reg_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            feat = self.backbone(x)
            feat = feat.view(feat.size(0), -1)
            return self.classifier(feat), self.reg_head(self.regressor_backbone(feat))

    return ImprovedMultimodelNet


def _reeval_nrmse(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    y_range = float(np.max(y_true) - np.min(y_true))
    if y_range < eps:
        return 0.0
    return float((rmse / y_range) * 100.0)


def _reeval_max_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.max(np.abs(y_true - y_pred)))


def save_single_model_feature_visualizations(
    backbone_features,
    regression_features,
    class_ids,
    true_wld,
    class_names,
    output_dir,
    title_prefix,
):
    """Save publication-quality (Q1/Q2 journal standard) PCA/t-SNE plots and composite figures."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    keep = np.arange(len(class_ids))
    if len(keep) > 750:
        keep = np.sort(rng.choice(keep, size=750, replace=False))
    class_ids = np.asarray(class_ids)[keep]
    true_wld = np.asarray(true_wld)[keep]

    # Distinct curated publication color palette
    pub_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    reg_cmaps = {"W": "viridis", "L": "plasma", "D": "turbo"}
    unit_map = {"W": "mm", "L": "mm", "D": "mm"}

    separation_rows = []
    projections_cache = {}

    for feature_key, feature_label, raw_features in [
        ("backbone", "Shared Backbone", backbone_features),
        ("regression", "Regression Embedding", regression_features),
    ]:
        features = np.asarray(raw_features)[keep]
        scaled = StandardScaler().fit_transform(features)
        pca_2d = PCA(n_components=2, random_state=42)
        pca_projection = pca_2d.fit_transform(scaled)
        pre_tsne_dim = min(50, scaled.shape[1], scaled.shape[0] - 1)
        pre_tsne = PCA(n_components=pre_tsne_dim, random_state=42).fit_transform(scaled)
        perplexity = min(30.0, max(5.0, (len(pre_tsne) - 1) / 3.0))
        tsne_signature = inspect.signature(TSNE)
        learning_rate_default = tsne_signature.parameters["learning_rate"].default
        tsne_kwargs = {
            "n_components": 2,
            "perplexity": perplexity,
            "learning_rate": (
                "auto" if isinstance(learning_rate_default, str) else 200.0
            ),
            "init": "pca",
            "random_state": 42,
            "method": "barnes_hut",
            "n_jobs": -1,
        }
        iteration_arg = (
            "max_iter"
            if "max_iter" in tsne_signature.parameters
            else "n_iter"
        )
        tsne_kwargs[iteration_arg] = 1000
        tsne_projection = TSNE(**tsne_kwargs).fit_transform(pre_tsne)

        if len(np.unique(class_ids)) > 1:
            silhouette = float(silhouette_score(scaled, class_ids))
            calinski = float(calinski_harabasz_score(scaled, class_ids))
            davies = float(davies_bouldin_score(scaled, class_ids))
        else:
            silhouette = calinski = davies = float("nan")

        separation_rows.append({
            "feature_space": feature_key,
            "feature_dim": int(features.shape[1]),
            "samples_visualized": int(len(features)),
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
            "pca_explained_variance_2d": float(
                np.sum(pca_2d.explained_variance_ratio_)
            ),
        })

        projections_cache[feature_key] = {
            "pca": pca_projection,
            "tsne": tsne_projection,
            "silhouette": silhouette,
            "davies": davies,
            "calinski": calinski,
            "pca_var": float(np.sum(pca_2d.explained_variance_ratio_)),
        }

        # --- Individual Large High-Res Figures ---
        for method_name, method_label, projection in [
            ("pca", "PCA", pca_projection),
            ("tsne", "t-SNE", tsne_projection),
        ]:
            fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
            for class_idx, class_name in enumerate(class_names):
                mask = class_ids == class_idx
                c_color = pub_colors[class_idx % len(pub_colors)]
                ax.scatter(
                    projection[mask, 0],
                    projection[mask, 1],
                    s=65,
                    alpha=0.82,
                    color=c_color,
                    edgecolors="black",
                    linewidth=0.35,
                    label=class_name,
                )
            ax.set_xlabel(f"{method_label} Dimension 1", fontsize=16, fontweight="bold")
            ax.set_ylabel(f"{method_label} Dimension 2", fontsize=16, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.set_title(
                f"{title_prefix}\n{feature_label} {method_label} by Crack Shape",
                fontsize=17,
                fontweight="bold",
                pad=12,
            )
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
            ax.legend(
                title="Crack Shape",
                title_fontsize=14,
                fontsize=13,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#cccccc",
                framealpha=0.95,
            )

            # Metric badge
            if np.isfinite(silhouette):
                metric_text = f"Silhouette (S): {silhouette:.3f}\nDBI: {davies:.3f}\nCH: {calinski:.1f}"
                if method_name == "pca":
                    metric_text += f"\nExpl. Var.: {projections_cache[feature_key]['pca_var']*100:.1f}%"
                ax.text(
                    0.03,
                    0.97,
                    metric_text,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        edgecolor="#888888",
                        alpha=0.92,
                    ),
                )

            fig.tight_layout()
            for ext in (".png", ".pdf"):
                fig.savefig(
                    os.path.join(output_dir, f"{feature_key}_{method_name}_by_class{ext}"),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)

            # Continuous Geometry Targets (W, L, D) for regression feature space
            if feature_key == "regression":
                for target_idx, target_name in enumerate(("W", "L", "D")):
                    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
                    cmap_choice = reg_cmaps.get(target_name, "viridis")
                    scatter = ax.scatter(
                        projection[:, 0],
                        projection[:, 1],
                        c=true_wld[:, target_idx],
                        cmap=cmap_choice,
                        s=70,
                        alpha=0.85,
                        edgecolors="black",
                        linewidth=0.3,
                    )
                    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
                    cbar.set_label(f"True {target_name} ({unit_map.get(target_name, 'mm')})", fontsize=15, fontweight="bold")
                    cbar.ax.tick_params(labelsize=13)
                    ax.set_xlabel(f"{method_label} Dimension 1", fontsize=16, fontweight="bold")
                    ax.set_ylabel(f"{method_label} Dimension 2", fontsize=16, fontweight="bold")
                    ax.tick_params(axis="both", which="major", labelsize=14)
                    ax.set_title(
                        f"{title_prefix}\nRegression Manifold {method_label} Colored by {target_name} ({unit_map.get(target_name, 'mm')})",
                        fontsize=17,
                        fontweight="bold",
                        pad=12,
                    )
                    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
                    fig.tight_layout()
                    for ext in (".png", ".pdf"):
                        fig.savefig(
                            os.path.join(
                                output_dir,
                                f"regression_{method_name}_colored_by_{target_name.lower()}{ext}",
                            ),
                            dpi=300,
                            bbox_inches="tight",
                        )
                    plt.close(fig)

    # --- COMPOSITE PUBLICATION FIGURES (Q1/Q2 Paper Standard) ---
    for method_name, method_label in [("tsne", "t-SNE"), ("pca", "PCA")]:
        # 1. Composite Figure: Backbone vs Regression Side-by-Side (1x2)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8.5), dpi=300)
        for sub_idx, (fkey, flabel) in enumerate([("backbone", "Shared Backbone"), ("regression", "Regression Embedding")]):
            ax = axes[sub_idx]
            proj = projections_cache[fkey][method_name]
            for class_idx, class_name in enumerate(class_names):
                mask = class_ids == class_idx
                c_color = pub_colors[class_idx % len(pub_colors)]
                ax.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    s=60,
                    alpha=0.82,
                    color=c_color,
                    edgecolors="black",
                    linewidth=0.35,
                    label=class_name,
                )
            ax.set_xlabel(f"{method_label} Dim 1", fontsize=16, fontweight="bold")
            ax.set_ylabel(f"{method_label} Dim 2", fontsize=16, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=14)
            sub_tag = f"({chr(97 + sub_idx)})"
            ax.set_title(f"{sub_tag} {flabel} Space", fontsize=17, fontweight="bold", pad=10)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)

            sil = projections_cache[fkey]["silhouette"]
            if np.isfinite(sil):
                ax.text(
                    0.03,
                    0.97,
                    f"Silhouette: {sil:.3f}\nDBI: {projections_cache[fkey]['davies']:.3f}\nCH: {projections_cache[fkey]['calinski']:.1f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#888888", alpha=0.92),
                )
            if sub_idx == 1:
                ax.legend(
                    title="Crack Shape",
                    title_fontsize=13,
                    fontsize=12,
                    loc="upper right",
                    frameon=True,
                    facecolor="white",
                    edgecolor="#cccccc",
                    framealpha=0.95,
                )

        fig.suptitle(f"{title_prefix} — {method_label} Latent Space Separation", fontsize=19, fontweight="bold", y=0.99)
        fig.tight_layout()
        for ext in (".png", ".pdf"):
            fig.savefig(
                os.path.join(output_dir, f"paper_composite_latent_{method_name}_2panel{ext}"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

        # 2. Composite Figure: 4-Panel Regression Physics Manifold (Shape, W, L, D) (2x2)
        reg_proj = projections_cache["regression"][method_name]
        fig, axes = plt.subplots(2, 2, figsize=(18, 15), dpi=300)

        # Panel (a): Shapes
        ax_shape = axes[0, 0]
        for class_idx, class_name in enumerate(class_names):
            mask = class_ids == class_idx
            c_color = pub_colors[class_idx % len(pub_colors)]
            ax_shape.scatter(
                reg_proj[mask, 0],
                reg_proj[mask, 1],
                s=60,
                alpha=0.82,
                color=c_color,
                edgecolors="black",
                linewidth=0.35,
                label=class_name,
            )
        ax_shape.set_title("(a) Shape Class Distribution", fontsize=16, fontweight="bold", pad=8)
        ax_shape.set_xlabel(f"{method_label} Dim 1", fontsize=14, fontweight="bold")
        ax_shape.set_ylabel(f"{method_label} Dim 2", fontsize=14, fontweight="bold")
        ax_shape.tick_params(axis="both", which="major", labelsize=13)
        ax_shape.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
        ax_shape.legend(title="Shape", title_fontsize=12, fontsize=11, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

        # Panels (b), (c), (d): Continuous W, L, D
        for t_idx, (t_name, t_ax, panel_label) in enumerate([
            ("W", axes[0, 1], "(b) Width $W$ (mm)"),
            ("L", axes[1, 0], "(c) Length $L$ (mm)"),
            ("D", axes[1, 1], "(d) Depth $D$ (mm)"),
        ]):
            cmap_choice = reg_cmaps.get(t_name, "viridis")
            sc = t_ax.scatter(
                reg_proj[:, 0],
                reg_proj[:, 1],
                c=true_wld[:, t_idx],
                cmap=cmap_choice,
                s=65,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.3,
            )
            cb = fig.colorbar(sc, ax=t_ax, fraction=0.046, pad=0.04)
            cb.set_label(f"True {t_name} ({unit_map[t_name]})", fontsize=13, fontweight="bold")
            cb.ax.tick_params(labelsize=12)
            t_ax.set_title(panel_label, fontsize=16, fontweight="bold", pad=8)
            t_ax.set_xlabel(f"{method_label} Dim 1", fontsize=14, fontweight="bold")
            t_ax.set_ylabel(f"{method_label} Dim 2", fontsize=14, fontweight="bold")
            t_ax.tick_params(axis="both", which="major", labelsize=13)
            t_ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)

        fig.suptitle(f"{title_prefix} — Physics Geometry Disentanglement ({method_label})", fontsize=19, fontweight="bold", y=0.99)
        fig.tight_layout()
        for ext in (".png", ".pdf"):
            fig.savefig(
                os.path.join(output_dir, f"paper_composite_geometry_manifold_{method_name}_4panel{ext}"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

    _atomic_to_csv(
        pd.DataFrame(separation_rows),
        os.path.join(output_dir, "feature_separation_metrics.csv"),
        index=False,
    )


def _reeval_load_base_data():
    global _REEVAL_BASE_DATA
    if _REEVAL_BASE_DATA is not None:
        return _REEVAL_BASE_DATA

    deps = _reeval_imports()
    load_data = deps["Load_Data_With_Labels"]
    X, y, matched_filenames = load_data(
        DATA_PATH,
        LABELS_PATH,
    )
    labels_df = pd.read_csv(LABELS_PATH)
    filename_to_shape = {}
    for _, row in labels_df.iterrows():
        try:
            filename = str(row["filename"]).strip()
            shape = str(row["shape"]).strip()
            if "type" in labels_df.columns and shape == "Step":
                type_val = str(row["type"]).strip()
                if type_val and type_val.lower() != "nan":
                    shape = f"Step_{type_val}"
            if shape and shape.lower() != "nan":
                filename_to_shape[filename] = shape
        except Exception:
            continue

    shape_text = []
    for filename in matched_filenames:
        shape_text.append(filename_to_shape.get(filename, str(filename).split("_")[0]))
    unique_shapes = sorted(set(shape_text))
    shape_to_idx = {shape: idx for idx, shape in enumerate(unique_shapes)}
    y_shape = np.array([shape_to_idx[shape] for shape in shape_text])

    _REEVAL_BASE_DATA = {
        "X": X,
        "y": y,
        "filenames": np.asarray(matched_filenames),
        "y_shape": y_shape,
        "unique_shapes": unique_shapes,
    }
    print(f"[REEVAL] Loaded base data: {len(X)} samples | shapes={unique_shapes}")
    return _REEVAL_BASE_DATA


def _reeval_prepare_split(percent, seed):
    key = (int(percent), int(seed))
    if key in _REEVAL_SPLIT_CACHE:
        return _REEVAL_SPLIT_CACHE[key]

    deps = _reeval_imports()
    train_test_split = deps["train_test_split"]
    StandardScaler = deps["StandardScaler"]
    DataLoader = deps["DataLoader"]
    TensorDataset = deps["TensorDataset"]
    torch = deps["torch"]

    base = _reeval_load_base_data()
    X = base["X"]
    y = base["y"]
    y_shape = base["y_shape"]
    filenames = base["filenames"]

    X_temp, X_test, y_temp, y_test, shape_temp, shape_test, fn_temp, fn_test = train_test_split(
        X, y, y_shape, filenames, test_size=0.10, random_state=int(seed)
    )
    X_train, X_val, y_train, y_val, shape_train, shape_val, fn_train, fn_val = train_test_split(
        X_temp, y_temp, shape_temp, fn_temp, test_size=0.222, random_state=int(seed)
    )

    subset_parts = []
    for class_idx in sorted(np.unique(shape_train)):
        class_idx = int(class_idx)
        class_mask_idx = np.where(shape_train == class_idx)[0]
        class_count = len(class_mask_idx)
        class_take = int(np.floor(class_count * (int(percent) / 100.0)))
        class_take = max(1, min(class_take, class_count))
        class_rng = np.random.default_rng(int(seed) + class_idx * 10007)
        subset_parts.append(np.sort(class_rng.permutation(class_mask_idx)[:class_take]))

    subset_indices = np.sort(np.concatenate(subset_parts))
    X_train = X_train[subset_indices]
    y_train = y_train[subset_indices]

    n_train, h, w, ch = X_train.shape
    n_test = len(X_test)
    scaler = StandardScaler()
    scaler.fit_transform(X_train.reshape(n_train * h * w, ch))
    X_test_norm = scaler.transform(X_test.reshape(n_test * h * w, ch)).reshape(n_test, h, w, ch)

    y_max = np.array([
        float(np.max(y_train[:, 0])),
        float(np.max(y_train[:, 1])),
        float(np.max(y_train[:, 2])),
    ])
    y_test_norm = y_test / y_max

    X_test_tensor = torch.FloatTensor(X_test_norm).permute(0, 3, 1, 2)
    y_test_tensor = torch.FloatTensor(y_test_norm)
    shape_test_tensor = torch.LongTensor(shape_test)
    loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor, shape_test_tensor),
        batch_size=max(1, BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    class _SeparateScaler:
        def __init__(self, max_vals):
            self.max_vals = max_vals
            self.data_max_ = max_vals
        def inverse_transform(self, X_norm):
            return np.asarray(X_norm) * self.max_vals
        def transform(self, X):
            return np.asarray(X) / self.max_vals

    split = {
        "loader": loader,
        "X_scaler": scaler,
        "y_scaler": _SeparateScaler(y_max),
        "y_test": y_test,
        "y_max": y_max,
        "shape_test": shape_test,
        "fn_test": fn_test,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "unique_shapes": base["unique_shapes"],
    }
    _REEVAL_SPLIT_CACHE[key] = split
    print(f"[REEVAL] Prepared split pct={percent}% seed={seed}: train={split['n_train']} val={split['n_val']} test={split['n_test']}")
    return split


def _reeval_checkpoint_candidates(run_dir, mode):
    custom = []
    eval_ckpt = os.environ.get('EVAL_CHECKPOINT', '').strip()
    if eval_ckpt:
        custom.append(eval_ckpt)

    best = [
        os.path.join(run_dir, "checkpoint_best_post_pinn", "best_model_pytorch.pth"),
        os.path.join(run_dir, "best_model_pytorch_post_pinn.pth"),
        os.path.join(run_dir, "checkpoint_best", "best_model_pytorch.pth"),
        os.path.join(run_dir, "best_model_pytorch.pth"),
    ]
    final = [
        os.path.join(run_dir, "checkpoint_final", "checkpoint_full.pth"),
        os.path.join(run_dir, "checkpoint_final", "model_final.pth"),
    ]
    latest = [
        os.path.join(run_dir, "checkpoint_latest", "checkpoint_full.pth"),
        os.path.join(run_dir, "checkpoint_latest", "model_latest.pth"),
    ]
    ordered = custom + best + final + latest
    return [path for path in ordered if os.path.exists(path)]


def _reeval_load_state(model, checkpoint_path, torch, device):
    try:
        obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)


def _save_reeval_fig3_regression_scatter(y_true, y_pred, metrics, output_dir, file_prefix=""):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics_names = ['Width ($W$, mm)', 'Length ($L$, mm)', 'Depth ($D$, mm)']
    colors = ['#004c6d', '#c35100', '#4a2c5d']
    
    for i in range(3):
        ax = axes3[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        
        ax.scatter(yt, yp, alpha=0.55, s=25, color=colors[i], edgecolors='white', linewidth=0.3, label='Predictions')
        
        min_v = min(float(yt.min()), float(yp.min()))
        max_v = max(float(yt.max()), float(yp.max()))
        ax.plot([min_v, max_v], [min_v, max_v], color='#d62728', linestyle='--', linewidth=1.8, label='Ideal ($y=x$)')
        
        mae = metrics['mae'][i]
        r2 = metrics['r2'][i]
        rmse = metrics['rmse'][i]
        nrmse = metrics['nrmse'][i]
        max_err = metrics.get('max_error', [np.nan, np.nan, np.nan])[i]
        
        metric_str = (
            f"MAE: {mae:.4f} mm\n"
            f"$R^2$: {r2:.4f}\n"
            f"RMSE: {rmse:.4f} mm\n"
            f"Max Err: {max_err:.4f} mm\n"
            f"NRMSE: {nrmse:.2f}%"
        )
        ax.text(0.05, 0.95, metric_str, transform=ax.transAxes, fontsize=8.5, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
        
        ax.set_xlabel(f'True {metrics_names[i]}', fontweight='bold')
        ax.set_ylabel(f'Predicted {metrics_names[i]}', fontweight='bold')
        ax.set_title(f'({chr(97+i)}) {metrics_names[i].split()[0]} Regression', fontweight='bold', pad=8)
        ax.legend(fontsize=8.5, loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
        
    plt.tight_layout()
    fname = f"{file_prefix}fig3_regression_scatter.png" if file_prefix else "fig3_regression_scatter.png"
    out_path = os.path.join(output_dir, fname)
    fig3.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    return out_path


def _save_reeval_fig2_confusion_matrix(y_shape_true, y_pred_shape, unique_shapes, output_dir, file_prefix=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    os.makedirs(output_dir, exist_ok=True)
    labels = list(range(len(unique_shapes)))
    cm = confusion_matrix(y_shape_true, y_pred_shape, labels=labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100.0
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=unique_shapes, yticklabels=unique_shapes,
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax2, annot_kws={'size': 10, 'weight': 'bold'})
    ax2.set_title('Test Set Confusion Matrix (%)', fontweight='bold', fontsize=12, pad=10)
    ax2.set_ylabel('True Crack Shape', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Predicted Crack Shape', fontweight='bold', fontsize=11)
    plt.tight_layout()
    fname = f"{file_prefix}fig2_confusion_matrix.png" if file_prefix else "fig2_confusion_matrix.png"
    out_path = os.path.join(output_dir, fname)
    fig2.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    return out_path


def _reeval_infer_existing_model(run_dir, mode, percent, seed, item):
    deps = _reeval_imports()
    torch = deps["torch"]
    nn = deps["nn"]
    mean_absolute_error = deps["mean_absolute_error"]
    mean_squared_error = deps["mean_squared_error"]
    median_absolute_error = deps["median_absolute_error"]
    max_error = deps["max_error"]
    explained_variance_score = deps["explained_variance_score"]
    r2_score = deps["r2_score"]
    balanced_accuracy_score = deps["balanced_accuracy_score"]
    cohen_kappa_score = deps["cohen_kappa_score"]
    matthews_corrcoef = deps["matthews_corrcoef"]
    precision_score = deps["precision_score"]
    recall_score = deps["recall_score"]
    f1_score = deps["f1_score"]
    precision_recall_fscore_support = deps["precision_recall_fscore_support"]
    confusion_matrix = deps["confusion_matrix"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split = _reeval_prepare_split(percent, seed)
    model_cls = _build_reeval_model_class(torch, nn)
    model = model_cls(len(split["unique_shapes"])).to(device)

    checkpoint_used = None
    load_errors = []
    for checkpoint_path in _reeval_checkpoint_candidates(run_dir, mode):
        try:
            _reeval_load_state(model, checkpoint_path, torch, device)
            checkpoint_used = checkpoint_path
            break
        except Exception as exc:
            load_errors.append(f"{checkpoint_path}: {exc}")
    if checkpoint_used is None:
        raise RuntimeError("No loadable checkpoint found. " + " | ".join(load_errors[:3]))

    model.eval()
    pred_shape_parts = []
    pred_wld_parts = []
    backbone_feature_parts = []
    regression_feature_parts = []
    with torch.no_grad():
        for batch_X, _, _ in split["loader"]:
            batch_X = batch_X.to(device, non_blocking=True)
            backbone_features = model.backbone(batch_X)
            backbone_features = backbone_features.view(backbone_features.size(0), -1)
            regression_features = model.regressor_backbone(backbone_features)
            logits = model.classifier(backbone_features)
            pred_wld = model.reg_head(regression_features)
            pred_shape_parts.append(torch.argmax(logits, dim=1).cpu().numpy())
            pred_wld_parts.append(pred_wld.cpu().numpy())
            backbone_feature_parts.append(backbone_features.cpu().numpy())
            regression_feature_parts.append(regression_features.cpu().numpy())

    y_pred_shape = np.concatenate(pred_shape_parts)
    y_pred_norm = np.concatenate(pred_wld_parts)
    backbone_features = np.concatenate(backbone_feature_parts)
    regression_features = np.concatenate(regression_feature_parts)
    y_pred = y_pred_norm * split["y_max"]
    y_true = split["y_test"]
    y_shape_true = split["shape_test"]
    unique_shapes = split["unique_shapes"]

    if not np.isfinite(y_pred).all():
        raise RuntimeError(f"Non-finite predictions from {checkpoint_used}")

    # Real experiment evaluation (Experiment_1)
    try:
        import pickle
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from load_real_experiment_data import evaluate_real_experiment
        x_sc = split.get("X_scaler")
        y_sc = split.get("y_scaler")
        x_sc_path = os.path.join(run_dir, "X_scaler.pkl")
        y_sc_path = os.path.join(run_dir, "y_scaler.pkl")
        if os.path.isfile(x_sc_path):
            try:
                with open(x_sc_path, "rb") as f:
                    x_sc = pickle.load(f)
            except Exception:
                pass
        if os.path.isfile(y_sc_path):
            try:
                with open(y_sc_path, "rb") as f:
                    y_sc = pickle.load(f)
            except Exception:
                pass
        evaluate_real_experiment(model, x_sc, y_sc, unique_shapes, run_dir, device=device)
    except Exception as exc:
        print(f"[WARN] Real experiment evaluation encountered an error for {run_dir}: {exc}")

    metrics = {
        "mae": [],
        "mse": [],
        "rmse": [],
        "r2": [],
        "median_ae": [],
        "max_error": [],
        "explained_variance": [],
        "mean_error": [],
        "nrmse": [],
        "nrmse_range": [],
    }
    for idx in range(3):
        true_target = y_true[:, idx]
        pred_target = y_pred[:, idx]
        residual = pred_target - true_target
        mse = mean_squared_error(true_target, pred_target)
        rmse = float(math.sqrt(mse))
        true_range = float(np.max(true_target) - np.min(true_target))
        nrmse_val = (rmse / true_range * 100.0) if true_range > 1e-12 else float("nan")
        metrics["mae"].append(float(mean_absolute_error(true_target, pred_target)))
        metrics["mse"].append(float(mse))
        metrics["rmse"].append(rmse)
        metrics["r2"].append(float(r2_score(true_target, pred_target)))
        metrics["median_ae"].append(float(median_absolute_error(true_target, pred_target)))
        metrics["max_error"].append(float(max_error(true_target, pred_target)))
        metrics["explained_variance"].append(float(explained_variance_score(true_target, pred_target)))
        metrics["mean_error"].append(float(np.mean(residual)))
        metrics["nrmse"].append(nrmse_val)
        metrics["nrmse_range"].append(nrmse_val)

    clf_acc = float(np.mean(y_pred_shape == y_shape_true))
    clf_metrics = {
        "accuracy": clf_acc,
        "balanced_accuracy": float(balanced_accuracy_score(y_shape_true, y_pred_shape)),
        "mcc": float(matthews_corrcoef(y_shape_true, y_pred_shape)),
        "cohen_kappa": float(cohen_kappa_score(y_shape_true, y_pred_shape)),
    }
    for average in ("macro", "micro", "weighted"):
        clf_metrics[f"precision_{average}"] = float(
            precision_score(y_shape_true, y_pred_shape, average=average, zero_division=0)
        )
        clf_metrics[f"recall_{average}"] = float(
            recall_score(y_shape_true, y_pred_shape, average=average, zero_division=0)
        )
        clf_metrics[f"f1_{average}"] = float(
            f1_score(y_shape_true, y_pred_shape, average=average, zero_division=0)
        )
    alpha, warmup = get_series_alpha_warmup(item)
    alpha = 0.0 if mode == "no_pinn" else alpha
    warmup = 0 if mode == "no_pinn" else warmup
    checkpoint_md5 = _file_md5(checkpoint_used)
    reeval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    common_meta = {
        "reeval_time": reeval_time,
        "series_key": item.get("key", ""),
        "series_label": item.get("label", ""),
        "series_short_label": item.get("short_label", ""),
        "series_tag": item.get("tag", ""),
        "series_group": item.get("group", ""),
        "deduped_from": item.get("_deduped_from", ""),
        "mode": mode,
        "train_percent": int(percent),
        "seed": str(seed),
        "epochs": EPOCHS,
        "pinn_loss_type": PINN_LOSS_TYPE,
        "pinn_alpha": alpha,
        "pinn_warmup": warmup,
        "model_arch": "ImprovedMultimodelNet",
        "model_num_shapes": len(unique_shapes),
        "model_shapes": "|".join(unique_shapes),
        "checkpoint_path": checkpoint_used,
        "checkpoint_file": os.path.basename(checkpoint_used),
        "checkpoint_md5": checkpoint_md5,
        "run_dir": run_dir,
        "reeval_batch_size": BATCH_SIZE,
        "reeval_train_samples": split["n_train"],
        "reeval_val_samples": split["n_val"],
        "reeval_test_samples": split["n_test"],
        "y_max_w": float(split["y_max"][0]),
        "y_max_l": float(split["y_max"][1]),
        "y_max_d": float(split["y_max"][2]),
    }

    result_row = {
        **common_meta,
        "clf_acc": clf_acc,
        **{f"clf_{key}": value for key, value in clf_metrics.items() if key != "accuracy"},
        "reeval_from_checkpoint": checkpoint_used,
    }
    target_suffixes = ("w", "l", "d")
    for metric_name, values in metrics.items():
        for suffix, value in zip(target_suffixes, values):
            result_row[f"{metric_name}_{suffix}"] = value
        result_row[f"{metric_name}_avg"] = float(np.nanmean(values))

    labels = list(range(len(unique_shapes)))
    prec, rec, f1, support = precision_recall_fscore_support(y_shape_true, y_pred_shape, labels=labels, zero_division=0)
    cm = confusion_matrix(y_shape_true, y_pred_shape, labels=labels)
    total = int(np.sum(cm))
    class_rows = []
    for cls_idx, class_name in enumerate(unique_shapes):
        tp = int(cm[cls_idx, cls_idx])
        fn = int(cm[cls_idx, :].sum() - tp)
        fp = int(cm[:, cls_idx].sum() - tp)
        tn = int(total - tp - fn - fp)
        n_samples = int(support[cls_idx])
        class_acc = float(tp / n_samples) if n_samples else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
        balanced_accuracy_ovr = 0.5 * (class_acc + specificity)
        mcc_denom = math.sqrt(
            float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
        mcc_ovr = float((tp * tn - fp * fn) / mcc_denom) if mcc_denom else 0.0
        class_rows.append({
            **common_meta,
            "class_index": cls_idx,
            "shape": class_name,
            "n_samples": n_samples,
            "n_correct": tp,
            "class_accuracy": class_acc,
            "class_accuracy_percent": class_acc * 100.0,
            "precision": float(prec[cls_idx]),
            "recall": float(rec[cls_idx]),
            "f1": float(f1[cls_idx]),
            "specificity": specificity,
            "balanced_accuracy_ovr": balanced_accuracy_ovr,
            "mcc_ovr": mcc_ovr,
            "one_vs_rest_accuracy": float((tp + tn) / total) if total else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "pinn_active": mode == "pinn",
        })

    reg_rows = []
    for idx, target in enumerate(["W", "L", "D"]):
        err = y_pred[:, idx] - y_true[:, idx]
        reg_rows.append({
            **common_meta,
            "target_index": idx,
            "target": target,
            "mae": metrics["mae"][idx],
            "mse": metrics["mse"][idx],
            "rmse": metrics["rmse"][idx],
            "r2": metrics["r2"][idx],
            "median_ae": metrics["median_ae"][idx],
            "max_error": metrics["max_error"][idx],
            "explained_variance": metrics["explained_variance"][idx],
            "mean_error": metrics["mean_error"][idx],
            "nrmse": metrics["nrmse"][idx],
            "nrmse_range": metrics["nrmse_range"][idx],
            "n_samples": int(len(y_true)),
            "true_mean": float(np.mean(y_true[:, idx])),
            "pred_mean": float(np.mean(y_pred[:, idx])),
            "error_mean": float(np.mean(err)),
            "error_std": float(np.std(err)),
            "pinn_active": mode == "pinn",
        })

    per_shape_rows = []
    for cls_idx, class_name in enumerate(unique_shapes):
        mask = y_shape_true == cls_idx
        if not np.any(mask):
            continue
        shape_mae = []
        shape_rmse = []
        shape_r2 = []
        shape_max_err = []
        shape_nrmse = []
        shape_bias = []
        for target_idx in range(3):
            mse = mean_squared_error(y_true[mask, target_idx], y_pred[mask, target_idx])
            shape_mae.append(float(mean_absolute_error(y_true[mask, target_idx], y_pred[mask, target_idx])))
            shape_rmse.append(float(math.sqrt(mse)))
            shape_r2.append(float(r2_score(y_true[mask, target_idx], y_pred[mask, target_idx])))
            shape_max_err.append(float(np.max(np.abs(y_true[mask, target_idx] - y_pred[mask, target_idx]))))
            shape_true_range = float(np.max(y_true[mask, target_idx]) - np.min(y_true[mask, target_idx]))
            shape_nrmse.append((math.sqrt(mse) / shape_true_range * 100.0) if shape_true_range > 1e-12 else float("nan"))
            shape_bias.append(float(np.mean(
                y_pred[mask, target_idx] - y_true[mask, target_idx]
            )))
        per_shape_rows.append({
            **common_meta,
            "shape": class_name,
            "n_samples": int(np.sum(mask)),
            "clf_acc": float(np.mean(y_pred_shape[mask] == y_shape_true[mask])),
            "mae_w": shape_mae[0],
            "mae_l": shape_mae[1],
            "mae_d": shape_mae[2],
            "mae_avg": float(np.mean(shape_mae)),
            "rmse_w": shape_rmse[0],
            "rmse_l": shape_rmse[1],
            "rmse_d": shape_rmse[2],
            "rmse_avg": float(np.mean(shape_rmse)),
            "r2_w": shape_r2[0],
            "r2_l": shape_r2[1],
            "r2_d": shape_r2[2],
            "r2_avg": float(np.mean(shape_r2)),
            "max_error_w": shape_max_err[0],
            "max_error_l": shape_max_err[1],
            "max_error_d": shape_max_err[2],
            "max_error_avg": float(np.mean(shape_max_err)),
            "nrmse_w": shape_nrmse[0],
            "nrmse_l": shape_nrmse[1],
            "nrmse_d": shape_nrmse[2],
            "nrmse_avg": float(np.nanmean(shape_nrmse)),
            "bias_w": shape_bias[0],
            "bias_l": shape_bias[1],
            "bias_d": shape_bias[2],
            "bias_avg": float(np.mean(shape_bias)),
            "pinn_active": mode == "pinn",
        })

    _atomic_to_csv(pd.DataFrame([result_row]), os.path.join(run_dir, "training_results.csv"), index=False)
    _atomic_to_csv(pd.DataFrame([result_row]), os.path.join(run_dir, "detailed_metrics.csv"), index=False)
    _atomic_to_csv(pd.DataFrame([result_row]), os.path.join(run_dir, "reeval_model_config.csv"), index=False)
    _atomic_to_csv(pd.DataFrame(class_rows), os.path.join(run_dir, "test_class_metrics.csv"), index=False)
    _atomic_to_csv(pd.DataFrame([{
        **common_meta,
        **{f"clf_{key}": value for key, value in clf_metrics.items()},
    }]), os.path.join(run_dir, "test_classification_metrics_overall.csv"), index=False)
    _atomic_to_csv(
        pd.DataFrame(cm, index=unique_shapes, columns=unique_shapes),
        os.path.join(run_dir, "test_confusion_matrix.csv"),
    )
    _atomic_to_csv(pd.DataFrame(reg_rows), os.path.join(run_dir, "test_regression_metrics_by_target.csv"), index=False)
    _atomic_to_csv(pd.DataFrame(per_shape_rows), os.path.join(run_dir, "test_metrics_per_shape.csv"), index=False)

    final_dir = os.path.join(run_dir, "checkpoint_final")
    os.makedirs(final_dir, exist_ok=True)
    _atomic_to_csv(pd.DataFrame([result_row]), os.path.join(final_dir, "test_results_final.csv"), index=False)
    _atomic_to_csv(pd.DataFrame([result_row]), os.path.join(final_dir, "test_detailed_final.csv"), index=False)
    _atomic_to_csv(pd.DataFrame([result_row]), os.path.join(final_dir, "reeval_model_config.csv"), index=False)
    _atomic_to_csv(pd.DataFrame(class_rows), os.path.join(final_dir, "test_class_metrics.csv"), index=False)
    _atomic_to_csv(pd.DataFrame([{
        **common_meta,
        **{f"clf_{key}": value for key, value in clf_metrics.items()},
    }]), os.path.join(final_dir, "test_classification_metrics_overall.csv"), index=False)
    _atomic_to_csv(
        pd.DataFrame(cm, index=unique_shapes, columns=unique_shapes),
        os.path.join(final_dir, "test_confusion_matrix.csv"),
    )
    _atomic_to_csv(pd.DataFrame(reg_rows), os.path.join(final_dir, "test_regression_metrics_by_target.csv"), index=False)
    _atomic_to_csv(pd.DataFrame(per_shape_rows), os.path.join(final_dir, "test_metrics_per_shape.csv"), index=False)

    if REEVAL_WRITE_PREDICTIONS:
        pred_rows = []
        for idx in range(len(y_shape_true)):
            true_w, true_l, true_d = y_true[idx]
            pred_w, pred_l, pred_d = y_pred[idx]
            pred_rows.append({
                **common_meta,
                "sample_index": idx,
                "filename": str(split["fn_test"][idx]),
                "true_shape": unique_shapes[int(y_shape_true[idx])],
                "pred_shape": unique_shapes[int(y_pred_shape[idx])],
                "shape_correct": bool(y_shape_true[idx] == y_pred_shape[idx]),
                "true_w": float(true_w),
                "true_l": float(true_l),
                "true_d": float(true_d),
                "pred_w": float(pred_w),
                "pred_l": float(pred_l),
                "pred_d": float(pred_d),
                "abs_err_w": float(abs(true_w - pred_w)),
                "abs_err_l": float(abs(true_l - pred_l)),
                "abs_err_d": float(abs(true_d - pred_d)),
            })
        pred_df = pd.DataFrame(pred_rows)
        _atomic_to_csv(pred_df, os.path.join(run_dir, "test_predictions.csv"), index=False)
        _atomic_to_csv(pred_df, os.path.join(final_dir, "test_predictions.csv"), index=False)

    feature_payload = {
        "backbone_features": backbone_features.astype(np.float32),
        "regression_features": regression_features.astype(np.float32),
        "true_class": y_shape_true.astype(np.int64),
        "pred_class": y_pred_shape.astype(np.int64),
        "true_wld": y_true.astype(np.float32),
        "pred_wld": y_pred.astype(np.float32),
        "filenames": np.asarray(split["fn_test"], dtype=str),
        "class_names": np.asarray(unique_shapes, dtype=str),
    }
    np.savez_compressed(os.path.join(run_dir, "test_feature_embeddings.npz"), **feature_payload)
    np.savez_compressed(os.path.join(final_dir, "test_feature_embeddings.npz"), **feature_payload)
    model_feature_dir = os.path.join(
        REPORT_DIR,
        "feature_visualization_by_model",
        _safe_filename(item.get("key", item.get("label", "model"))),
        f"train_{int(percent):02d}pct_seed_{seed}",
    )
    save_single_model_feature_visualizations(
        backbone_features=backbone_features,
        regression_features=regression_features,
        class_ids=y_shape_true,
        true_wld=y_true,
        class_names=unique_shapes,
        output_dir=model_feature_dir,
        title_prefix=(
            f"{item.get('label', item.get('key', 'model'))} | "
            f"train={percent}% | seed={seed}"
        ),
    )

    # Re-plot & save IEEE scatter (fig3) and confusion matrix (fig2)
    _save_reeval_fig3_regression_scatter(y_true, y_pred, metrics, run_dir)
    _save_reeval_fig3_regression_scatter(y_true, y_pred, metrics, final_dir)
    _save_reeval_fig2_confusion_matrix(y_shape_true, y_pred_shape, unique_shapes, run_dir)
    _save_reeval_fig2_confusion_matrix(y_shape_true, y_pred_shape, unique_shapes, final_dir)

    # Save to master evaluation directory (_eval / REPORT_DIR)
    by_pct_dir = os.path.join(REPORT_DIR, "by_percent", f"train_{int(percent):02d}pct")
    paper_reg_dir = os.path.join(REPORT_DIR, "paper_plots", "regression_plots")
    paper_clf_dir = os.path.join(REPORT_DIR, "paper_plots", "classification_plots")
    series_tag = item.get("tag", f"{mode}_a{alpha}_W{warmup}")
    
    _save_reeval_fig3_regression_scatter(y_true, y_pred, metrics, by_pct_dir, file_prefix=f"{mode}_{series_tag}_seed_{seed}_")
    _save_reeval_fig3_regression_scatter(y_true, y_pred, metrics, paper_reg_dir, file_prefix=f"fig3_{mode}_{series_tag}_seed_{seed}_")
    _save_reeval_fig2_confusion_matrix(y_shape_true, y_pred_shape, unique_shapes, by_pct_dir, file_prefix=f"{mode}_{series_tag}_seed_{seed}_")
    _save_reeval_fig2_confusion_matrix(y_shape_true, y_pred_shape, unique_shapes, paper_clf_dir, file_prefix=f"fig2_{mode}_{series_tag}_seed_{seed}_")

    config_txt = "\n".join([
        "REEVALUATION MODEL CONFIG",
        "=" * 80,
        f"reeval_time: {reeval_time}",
        f"series_key: {common_meta['series_key']}",
        f"series_label: {common_meta['series_label']}",
        f"series_tag: {common_meta['series_tag']}",
        f"mode: {mode}",
        f"train_percent: {percent}",
        f"seed: {seed}",
        f"epochs: {EPOCHS}",
        f"pinn_loss_type: {PINN_LOSS_TYPE}",
        f"pinn_alpha: {alpha}",
        f"pinn_warmup: {warmup}",
        f"model_arch: {common_meta['model_arch']}",
        f"model_shapes: {common_meta['model_shapes']}",
        f"checkpoint_path: {checkpoint_used}",
        f"checkpoint_md5: {checkpoint_md5}",
        f"run_dir: {run_dir}",
        "",
        "REEVALUATION SPLIT",
        "=" * 80,
        f"train_samples_after_percent: {split['n_train']}",
        f"val_samples: {split['n_val']}",
        f"test_samples: {split['n_test']}",
        f"y_max_w: {split['y_max'][0]}",
        f"y_max_l: {split['y_max'][1]}",
        f"y_max_d: {split['y_max'][2]}",
        "",
        "RESULTS",
        "=" * 80,
        f"clf_acc: {clf_acc}",
        f"clf_balanced_accuracy: {clf_metrics['balanced_accuracy']}",
        f"clf_precision_macro: {clf_metrics['precision_macro']}",
        f"clf_recall_macro: {clf_metrics['recall_macro']}",
        f"clf_f1_macro: {clf_metrics['f1_macro']}",
        f"clf_precision_weighted: {clf_metrics['precision_weighted']}",
        f"clf_recall_weighted: {clf_metrics['recall_weighted']}",
        f"clf_f1_weighted: {clf_metrics['f1_weighted']}",
        f"clf_mcc: {clf_metrics['mcc']}",
        f"clf_cohen_kappa: {clf_metrics['cohen_kappa']}",
        f"mae_avg: {result_row['mae_avg']}",
        f"rmse_avg: {result_row['rmse_avg']}",
        f"r2_avg: {result_row['r2_avg']}",
        f"max_error_avg: {result_row['max_error_avg']}",
        f"nrmse_avg: {result_row['nrmse_avg']}",
        f"mae_w: {result_row['mae_w']}",
        f"mae_l: {result_row['mae_l']}",
        f"mae_d: {result_row['mae_d']}",
        f"rmse_w: {result_row['rmse_w']}",
        f"rmse_l: {result_row['rmse_l']}",
        f"rmse_d: {result_row['rmse_d']}",
        f"r2_w: {result_row['r2_w']}",
        f"r2_l: {result_row['r2_l']}",
        f"r2_d: {result_row['r2_d']}",
        f"nrmse_w: {result_row['nrmse_w']}",
        f"nrmse_l: {result_row['nrmse_l']}",
        f"nrmse_d: {result_row['nrmse_d']}",
        "",
    ])
    _atomic_to_text(config_txt, os.path.join(run_dir, "reeval_model_config.txt"))
    _atomic_to_text(config_txt, os.path.join(final_dir, "reeval_model_config.txt"))

    print(
        f"[REEVAL][OK] pct={percent}% seed={seed} {item['label']} "
        f"acc={clf_acc*100.0:.2f}% bal_acc={clf_metrics['balanced_accuracy']*100.0:.2f}% "
        f"mae={result_row['mae_avg']:.4f} rmse={result_row['rmse_avg']:.4f} "
        f"r2={result_row['r2_avg']:.4f} nrmse={result_row['nrmse_avg']:.2f}%"
    )
    return result_row


def _csv_has_required_data(
    path,
    required_columns,
    *,
    min_rows=1,
    entity_column=None,
    required_entities=None,
):
    if not os.path.isfile(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if len(df) < min_rows or not set(required_columns).issubset(df.columns):
        return False
    if entity_column and required_entities:
        if entity_column not in df.columns:
            return False
        actual = set(df[entity_column].dropna().astype(str))
        if not set(required_entities).issubset(actual):
            return False
    numeric = df[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def _missing_metric_groups(run_dir, item, percent, seed):
    """Return metric groups that are absent or incomplete for one model run."""
    missing = []
    overall_classification = {
        "clf_accuracy", "clf_balanced_accuracy",
        "clf_precision_macro", "clf_recall_macro", "clf_f1_macro",
        "clf_precision_micro", "clf_recall_micro", "clf_f1_micro",
        "clf_precision_weighted", "clf_recall_weighted", "clf_f1_weighted",
        "clf_mcc", "clf_cohen_kappa",
    }
    per_class = {
        "class_accuracy", "class_accuracy_percent", "precision", "recall",
        "f1", "specificity", "balanced_accuracy_ovr", "mcc_ovr",
        "one_vs_rest_accuracy", "tp", "fp", "fn", "tn", "n_samples",
    }
    regression_target = {
        "mae", "mse", "rmse", "r2", "median_ae", "max_error",
        "explained_variance", "mean_error", "nrmse", "nrmse_range",
        "error_mean", "error_std", "true_mean", "pred_mean", "n_samples",
    }
    regression_overall = {
        f"{metric}_avg"
        for metric in (
            "mae", "mse", "rmse", "r2", "median_ae", "max_error",
            "explained_variance", "mean_error", "nrmse", "nrmse_range",
        )
    }
    regression_shape = {
        f"{metric}_{suffix}"
        for metric in ("mae", "rmse", "r2", "max_error", "nrmse", "bias")
        for suffix in ("w", "l", "d")
    } | {"nrmse_avg", "bias_avg", "clf_acc", "n_samples"}

    checks = [
        (
            "classification_overall",
            os.path.join(run_dir, "test_classification_metrics_overall.csv"),
            overall_classification,
            1,
            None,
            None,
        ),
        (
            "classification_per_class",
            os.path.join(run_dir, "test_class_metrics.csv"),
            per_class,
            5,
            "shape",
            ["Ellipse", "Rectangular", "Step_R", "Step_T", "Triangular"],
        ),
        (
            "regression_overall",
            os.path.join(run_dir, "training_results.csv"),
            regression_overall,
            1,
            None,
            None,
        ),
        (
            "regression_per_target",
            os.path.join(run_dir, "test_regression_metrics_by_target.csv"),
            regression_target,
            3,
            "target",
            ["W", "L", "D"],
        ),
        (
            "regression_per_shape",
            os.path.join(run_dir, "test_metrics_per_shape.csv"),
            regression_shape,
            5,
            "shape",
            ["Ellipse", "Rectangular", "Step_R", "Step_T", "Triangular"],
        ),
        (
            "real_experiment",
            os.path.join(run_dir, "real_experiment_results", "real_experiment_summary_metrics.csv"),
            {"Split", "Frequency", "Num_Samples"},
            1,
            None,
            None,
        ),
    ]
    for group, path, columns, min_rows, entity_column, entities in checks:
        if not _csv_has_required_data(
            path,
            columns,
            min_rows=min_rows,
            entity_column=entity_column,
            required_entities=entities,
        ):
            missing.append(group)

    feature_path = os.path.join(
        REPORT_DIR,
        "feature_visualization_by_model",
        _safe_filename(item.get("key", item.get("label", "model"))),
        f"train_{int(percent):02d}pct_seed_{seed}",
        "feature_separation_metrics.csv",
    )
    if not _csv_has_required_data(
        feature_path,
        {
            "feature_dim", "samples_visualized", "silhouette",
            "calinski_harabasz", "davies_bouldin",
            "pca_explained_variance_2d",
        },
        min_rows=2,
        entity_column="feature_space",
        required_entities=["backbone", "regression"],
    ):
        missing.append("feature_space")
    return missing


def _recover_feature_metrics_from_embeddings(run_dir, item, percent, seed):
    feature_output_dir = os.path.join(
        REPORT_DIR,
        "feature_visualization_by_model",
        _safe_filename(item.get("key", item.get("label", "model"))),
        f"train_{int(percent):02d}pct_seed_{seed}",
    )
    feature_metrics_path = os.path.join(
        feature_output_dir,
        "feature_separation_metrics.csv",
    )
    embedding_path = next(
        (
            path for path in (
                os.path.join(run_dir, "test_feature_embeddings.npz"),
                os.path.join(
                    run_dir,
                    "checkpoint_final",
                    "test_feature_embeddings.npz",
                ),
            )
            if os.path.isfile(path)
        ),
        None,
    )
    if embedding_path is None:
        return False
    try:
        with np.load(embedding_path, allow_pickle=False) as data:
            save_single_model_feature_visualizations(
                backbone_features=data["backbone_features"],
                regression_features=data["regression_features"],
                class_ids=data["true_class"],
                true_wld=data["true_wld"],
                class_names=data["class_names"].astype(str),
                output_dir=feature_output_dir,
                title_prefix=(
                    f"{item.get('label', item.get('key', 'model'))} | "
                    f"train={percent}% | seed={seed}"
                ),
            )
        print(
            "[FEATURE][RECOVER] Rebuilt feature metrics from saved embeddings: "
            f"{feature_metrics_path}"
        )
        return os.path.isfile(feature_metrics_path)
    except Exception as exc:
        print(
            f"[FEATURE][WARN] Recovery failed for {item.get('key')} "
            f"pct={percent} seed={seed}: {exc}"
        )
        return False


def reevaluate_existing_models_for_series(series_items):
    if not REEVAL_IF_MODEL_EXISTS:
        return []
    print("\n[REEVAL] Checking existing model checkpoints and refreshing CSV metrics...")
    reeval_dir = os.path.join(REPORT_DIR, "reeval_existing")
    os.makedirs(reeval_dir, exist_ok=True)
    summary_path = os.path.join(reeval_dir, "reeval_summary.csv")
    failures_path = os.path.join(reeval_dir, "reeval_failures.csv")
    rows = (
        pd.read_csv(summary_path).to_dict("records")
        if os.path.isfile(summary_path) else []
    )
    failures = (
        pd.read_csv(failures_path).to_dict("records")
        if os.path.isfile(failures_path) else []
    )
    seen = set()
    for item in series_items:
        for pct in item.get("percent_list", PERCENT_LIST):
            for seed in item.get("seed_list", SEEDS):
                seed_suffix = f"{item['tag']}_seed_{seed}"
                run_dir = find_latest_run_dir(get_percent_dir(item["mode"], pct), seed_suffix)
                key = (
                    item["mode"],
                    int(pct),
                    str(seed),
                    os.path.normpath(run_dir) if run_dir else "",
                )
                if not run_dir or key in seen:
                    continue
                seen.add(key)
                if not _reeval_checkpoint_candidates(run_dir, item["mode"]):
                    continue
                missing_groups = _missing_metric_groups(
                    run_dir,
                    item,
                    int(pct),
                    str(seed),
                )
                if missing_groups == ["feature_space"]:
                    _recover_feature_metrics_from_embeddings(
                        run_dir,
                        item,
                        int(pct),
                        str(seed),
                    )
                    missing_groups = _missing_metric_groups(
                        run_dir,
                        item,
                        int(pct),
                        str(seed),
                    )
                if not missing_groups and not FORCE_REEVAL:
                    _REEVAL_SUCCESS_RUN_DIRS.add(os.path.normpath(run_dir))
                    print(
                        f"[REEVAL][SKIP] Complete metrics: {item['label']} "
                        f"pct={pct}% seed={seed}"
                    )
                    continue
                print(
                    f"[REEVAL][MISSING] {item['label']} pct={pct}% seed={seed}: "
                    f"{', '.join(missing_groups)}"
                )
                try:
                    row = _reeval_infer_existing_model(run_dir, item["mode"], int(pct), int(seed), item)
                    _REEVAL_SUCCESS_RUN_DIRS.add(os.path.normpath(run_dir))
                    row["run_dir"] = run_dir
                    row["series"] = item["label"]
                    rows = [
                        existing
                        for existing in rows
                        if os.path.normpath(str(existing.get("run_dir", "")))
                        != os.path.normpath(run_dir)
                    ]
                    rows.append(row)
                    _atomic_to_csv(
                        pd.DataFrame(rows),
                        summary_path,
                        index=False,
                    )
                    save_all_metrics_long_csv(series_items, REPORT_DIR)
                    try:
                        refresh_live_comparison_plots(series_items, REPORT_DIR)
                    except Exception as refresh_exc:
                        print(
                            f"[LIVE][WARN] Incremental plot refresh failed after "
                            f"{item['label']} pct={pct}% seed={seed}: {refresh_exc}"
                        )
                    print(
                        f"[REEVAL][LIVE] Updated report after {item['label']} "
                        f"pct={pct}% seed={seed}: {REPORT_DIR}"
                    )
                except Exception as exc:
                    failures.append({"series": item["label"], "percent": pct, "seed": seed, "run_dir": run_dir, "error": repr(exc)})
                    _atomic_to_csv(
                        pd.DataFrame(failures),
                        failures_path,
                        index=False,
                    )
                    print(f"[REEVAL][WARN] Failed {item['label']} pct={pct}% seed={seed}: {exc}")

    if rows:
        _atomic_to_csv(pd.DataFrame(rows), summary_path, index=False)
        print(f"[REEVAL][OK] Refreshed {len(rows)} existing run(s).")
    else:
        print("[REEVAL] No existing checkpoints refreshed.")
    if failures:
        _atomic_to_csv(pd.DataFrame(failures), failures_path, index=False)
        print(f"[REEVAL][WARN] Failures saved: {failures_path}")
    return rows



def find_missing_training_units(series_items):
    """Return configured runs that do not have a completed checkpoint yet."""
    missing = []
    for item in series_items:
        for pct in item.get("percent_list", PERCENT_LIST):
            for seed in item.get("seed_list", SEEDS):
                seed_suffix = f"{item['tag']}_seed_{seed}"
                run_dir = find_latest_run_dir(
                    get_percent_dir(item["mode"], pct), seed_suffix
                )
                if not run_dir or not has_completed_checkpoint(run_dir):
                    missing.append(
                        {
                            "series_key": item.get("key", ""),
                            "label": item.get("label", ""),
                            "mode": item.get("mode", ""),
                            "train_percent": int(pct),
                            "seed": str(seed),
                            "tag": seed_suffix,
                            "latest_run_dir": run_dir or "",
                        }
                    )
    return missing


def preflight_infer_plot_existing(series_items):
    """Audit configured runs, report missing training units, and prepare training scheduler."""
    print("\n" + "=" * 80)
    print("BƯỚC 1: KIỂM TRA TRẠNG THÁI CÁC TRƯỜNG HỢP HUẤN LUYỆN (AUDIT CHECKPOINTS)")
    print("=" * 80)

    missing = find_missing_training_units(series_items)
    if missing:
        missing_path = os.path.join(REPORT_DIR, "missing_training_units_before_train.csv")
        _atomic_to_csv(pd.DataFrame(missing), missing_path, index=False)
        print(
            f"[AUDIT] Phát hiện {len(missing)} trường hợp CHƯA hoàn thành huấn luyện (cần train):"
        )
        for row in missing:
            print(
                f"  - [CẦN TRAIN] {row['label']} | Train: {row['train_percent']}% | Seed: {row['seed']}"
            )
        print(f"[AUDIT] Danh sách lưu tại: {missing_path}")
        print("[AUDIT] Hệ thống sẽ tự động kích hoạt huấn luyện cho các trường hợp còn thiếu trước khi đánh giá.")
    else:
        print("[AUDIT] Tất cả các cấu hình/tỷ lệ dữ liệu/seed đã hoàn thành huấn luyện (ĐẦY ĐỦ CHECKPOINTS).")
        print("[AUDIT] Sẵn sàng chuyển sang bước Đánh giá Suy luận & Đồ thị Tổng hợp.")

    if INFER_BEFORE_TRAIN:
        print("\n[PREFLIGHT] INFER_BEFORE_TRAIN=1 -> Chạy làm mới suy luận sơ bộ cho các checkpoint sẵn có...")
        reevaluate_existing_models_for_series(series_items)
        try:
            save_all_metrics_long_csv(series_items, REPORT_DIR)
            collect_series_results(series_items)
            if LIVE_PROGRESS_PLOTS:
                refresh_live_comparison_plots(series_items, REPORT_DIR)
        except Exception as exc:
            print(f"[PREFLIGHT][WARN] Chưa thể cập nhật biểu đồ sơ bộ: {exc}")

    return missing

def save_latent_analysis_tables(metrics_long_df, report_dir):
    """Save latent-space separation tables and plots for backbone/regression."""
    latent_dir = os.path.join(report_dir, "latent_analysis")
    os.makedirs(latent_dir, exist_ok=True)

    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    latent = metrics_long_df[
        (metrics_long_df["scope"] == "feature_space")
        & (metrics_long_df["entity_name"].isin(["backbone", "regression"]))
        & (metrics_long_df["metric"].isin(metrics))
    ].copy()
    if latent.empty:
        return []

    index_cols = [
        "model_id", "model_name", "mode", "alpha", "warmup",
        "train_percent", "seed", "entity_name",
    ]
    table = latent.pivot_table(
        index=index_cols,
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    table = table.rename(columns={
        "entity_name": "feature_space",
        "silhouette": "silhouette_higher_better",
        "davies_bouldin": "davies_bouldin_lower_better",
        "calinski_harabasz": "calinski_harabasz_higher_better",
    })

    baseline = table[table["mode"] == "no_pinn"][
        [
            "train_percent", "seed", "feature_space",
            "silhouette_higher_better",
            "davies_bouldin_lower_better",
            "calinski_harabasz_higher_better",
        ]
    ].drop_duplicates(["train_percent", "seed", "feature_space"])
    baseline = baseline.rename(columns={
        "silhouette_higher_better": "baseline_silhouette",
        "davies_bouldin_lower_better": "baseline_davies_bouldin",
        "calinski_harabasz_higher_better": "baseline_calinski_harabasz",
    })
    table = table.merge(
        baseline,
        on=["train_percent", "seed", "feature_space"],
        how="left",
    )
    table["silhouette_improvement_vs_no_pinn"] = (
        table["silhouette_higher_better"] - table["baseline_silhouette"]
    )
    table["davies_bouldin_improvement_vs_no_pinn"] = (
        table["baseline_davies_bouldin"]
        - table["davies_bouldin_lower_better"]
    )
    table["calinski_harabasz_improvement_vs_no_pinn"] = (
        table["calinski_harabasz_higher_better"]
        - table["baseline_calinski_harabasz"]
    )
    table = table.sort_values(
        ["feature_space", "train_percent", "seed", "mode", "alpha", "warmup"],
        kind="stable",
    )

    csv_path = os.path.join(latent_dir, "latent_separation_metrics.csv")
    _atomic_to_csv(table, csv_path, index=False)
    saved = [csv_path]

    display_names = {
        "silhouette_higher_better": "Silhouette ↑",
        "davies_bouldin_lower_better": "DBI ↓",
        "calinski_harabasz_higher_better": "CH ↑",
    }
    metric_columns = list(display_names)

    for feature_space in ("backbone", "regression"):
        feature_df = table[table["feature_space"] == feature_space].copy()
        if feature_df.empty:
            continue
        feature_df["model_display"] = feature_df.apply(
            lambda row: (
                f"{row['model_name']} | {int(row['train_percent'])}%"
                f" | seed={row['seed']}"
            ),
            axis=1,
        )

        raw = feature_df.set_index("model_display")[metric_columns]
        colors = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
        for column in metric_columns:
            values = raw[column].astype(float)
            vmin, vmax = float(values.min()), float(values.max())
            normalized = (
                pd.Series(0.5, index=values.index)
                if math.isclose(vmin, vmax)
                else (values - vmin) / (vmax - vmin)
            )
            if column == "davies_bouldin_lower_better":
                normalized = 1.0 - normalized
            colors[column] = normalized

        annotations = raw.copy().astype(object)
        annotations["silhouette_higher_better"] = raw[
            "silhouette_higher_better"
        ].map(lambda value: f"{value:.4f}")
        annotations["davies_bouldin_lower_better"] = raw[
            "davies_bouldin_lower_better"
        ].map(lambda value: f"{value:.4f}")
        annotations["calinski_harabasz_higher_better"] = raw[
            "calinski_harabasz_higher_better"
        ].map(lambda value: f"{value:.2f}")
        colors = colors.rename(columns=display_names)
        annotations = annotations.rename(columns=display_names)

        fig_height = max(6.0, 1.8 + 0.48 * len(raw))
        fig, ax = plt.subplots(figsize=(11, fig_height))
        sns.heatmap(
            colors,
            annot=annotations,
            fmt="",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(
            f"Latent separation metrics - {feature_space.title()}\n"
            "Silhouette/CH: higher is better; DBI: lower is better",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        fig.tight_layout()
        heatmap_path = os.path.join(
            latent_dir,
            f"latent_separation_{feature_space}_table.png",
        )
        fig.savefig(heatmap_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved.append(heatmap_path)

        for metric_column, metric_label in display_names.items():
            plot_df = feature_df.sort_values(
                ["train_percent", "seed", "mode", "alpha", "warmup"],
                kind="stable",
            )
            fig_width = max(12.0, 0.55 * len(plot_df))
            fig, ax = plt.subplots(figsize=(fig_width, 6.5))
            bar_colors = [
                "#2878B5" if mode == "no_pinn" else "#E07B39"
                for mode in plot_df["mode"]
            ]
            ax.bar(
                np.arange(len(plot_df)),
                plot_df[metric_column],
                color=bar_colors,
                alpha=0.88,
            )
            ax.set_xticks(np.arange(len(plot_df)))
            ax.set_xticklabels(
                plot_df["model_display"],
                rotation=55,
                ha="right",
                fontsize=8,
            )
            ax.set_ylabel(metric_label)
            ax.set_title(
                f"{feature_space.title()} latent space - {metric_label}",
                fontweight="bold",
            )
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            plot_path = os.path.join(
                latent_dir,
                f"latent_{feature_space}_{metric_column}.png",
            )
            fig.savefig(plot_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            saved.append(plot_path)

    print(f"[OK] Saved latent analysis: {latent_dir}")
    return saved


def save_full_metric_tables(metrics_long_df, report_dir):
    """Render readable, paginated heatmap tables for every collected metric."""
    if metrics_long_df.empty:
        return []

    output_dir = os.path.join(report_dir, "full_metric_tables")
    os.makedirs(output_dir, exist_ok=True)
    save_latent_analysis_tables(metrics_long_df, report_dir)

    df = metrics_long_df.copy()
    df["model_display"] = df.apply(
        lambda row: (
            f"{row['model_name']} | {int(row['train_percent'])}%"
            f" | seed={row['seed']}"
        ),
        axis=1,
    )

    key_metrics = {
        ("classification_overall", "overall"): [
            "clf_balanced_accuracy", "clf_precision_macro",
            "clf_recall_macro", "clf_f1_macro", "clf_mcc",
        ],
        ("regression_overall", "average_wld"): [
            "mae_avg", "rmse_avg", "r2_avg", "max_error_avg", "nrmse_avg",
        ],
    }
    overview_parts = []
    for (scope, entity_name), metrics in key_metrics.items():
        part = df[
            (df["scope"] == scope)
            & (df["entity_name"] == entity_name)
            & (df["metric"].isin(metrics))
        ].copy()
        overview_parts.append(part)
    overview_df = pd.concat(overview_parts, ignore_index=True)
    overview_csv = os.path.join(output_dir, "main_metrics_overview.csv")
    overview_wide = overview_df.pivot_table(
        index=[
            "model_id", "model_name", "mode", "alpha", "warmup",
            "train_percent", "seed",
        ],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    _atomic_to_csv(overview_wide, overview_csv, index=False)

    saved_paths = []

    def render_table(raw_table, direction_by_column, title, path):
        raw_table = raw_table.dropna(axis=1, how="all")
        if raw_table.empty:
            return

        color_table = raw_table.copy()
        annotations = raw_table.copy().astype(object)
        for column in raw_table.columns:
            values = pd.to_numeric(raw_table[column], errors="coerce")
            finite = values[np.isfinite(values)]
            if finite.empty:
                color_table[column] = np.nan
                annotations[column] = ""
                continue

            vmin = float(finite.min())
            vmax = float(finite.max())
            if math.isclose(vmin, vmax):
                normalized = pd.Series(0.5, index=values.index)
            else:
                normalized = (values - vmin) / (vmax - vmin)

            direction = direction_by_column.get(column, "higher")
            if direction == "lower":
                normalized = 1.0 - normalized
            elif direction == "closer_to_zero":
                absolute = values.abs()
                amin = float(absolute[np.isfinite(absolute)].min())
                amax = float(absolute[np.isfinite(absolute)].max())
                normalized = (
                    pd.Series(0.5, index=values.index)
                    if math.isclose(amin, amax)
                    else 1.0 - ((absolute - amin) / (amax - amin))
                )
            color_table[column] = normalized

            def format_value(value):
                if pd.isna(value):
                    return ""
                if abs(float(value)) >= 1000:
                    return f"{float(value):.0f}"
                if abs(float(value)) >= 100:
                    return f"{float(value):.2f}"
                return f"{float(value):.4f}"

            annotations[column] = values.map(format_value)

        fig_width = max(12.0, 2.0 + 1.65 * len(raw_table.columns))
        fig_height = max(5.0, 1.8 + 0.48 * len(raw_table.index))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            color_table.astype(float),
            annot=annotations,
            fmt="",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.4,
            linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=38, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    overview_plot = overview_df.copy()
    overview_plot["column_name"] = overview_plot["metric_label"]
    overview_table = overview_plot.pivot_table(
        index="model_display",
        columns="column_name",
        values="value",
        aggfunc="first",
    )
    overview_direction = (
        overview_plot.drop_duplicates("column_name")
        .set_index("column_name")["better_direction"]
        .to_dict()
    )
    render_table(
        overview_table,
        overview_direction,
        "Main metrics by model (green = better within each column)",
        os.path.join(output_dir, "main_metrics_overview.png"),
    )

    max_columns = 10
    for scope in sorted(df["scope"].dropna().unique()):
        scope_df = df[df["scope"] == scope].copy()
        scope_df["column_name"] = scope_df.apply(
            lambda row: (
                str(row["metric_label"])
                if str(row["entity_name"]) in {"overall", "average_wld"}
                else f"{row['entity_name']} | {row['metric_label']}"
            ),
            axis=1,
        )
        column_order = list(dict.fromkeys(scope_df["column_name"].tolist()))
        for page_index, start in enumerate(range(0, len(column_order), max_columns), 1):
            page_columns = column_order[start:start + max_columns]
            page_df = scope_df[scope_df["column_name"].isin(page_columns)]
            table = page_df.pivot_table(
                index="model_display",
                columns="column_name",
                values="value",
                aggfunc="first",
            ).reindex(columns=page_columns)
            directions = (
                page_df.drop_duplicates("column_name")
                .set_index("column_name")["better_direction"]
                .to_dict()
            )
            render_table(
                table,
                directions,
                f"{scope.replace('_', ' ').title()} - page {page_index}",
                os.path.join(
                    output_dir,
                    f"{_safe_filename(scope)}_page_{page_index:02d}.png",
                ),
            )

    print(
        f"[OK] Saved full metric tables: {output_dir} "
        f"({len(saved_paths)} images)"
    )
    return saved_paths


def save_all_metrics_long_csv(series_items, report_dir):
    """Save model metrics as values, without embedding filesystem paths."""
    rows = []
    seen = set()

    def metric_metadata(scope, metric):
        if metric in {"nrmse", "nrmse_range", "nrmse_mean"} or metric.startswith(("nrmse_", "nrmse_range_", "nrmse_mean_")):
            return "NRMSE (%)", "lower", "%"
        if metric in {"mean_error", "error_mean"} or metric.startswith(("mean_error_", "bias_")):
            return "Bias" if metric in {"mean_error", "error_mean"} else metric.replace("mean_error", "bias"), "closer_to_zero", "mm"
        if metric in {"mae", "mse", "rmse", "median_ae", "max_error"} or metric.startswith(
            ("mae_", "mse_", "rmse_", "median_ae_", "max_error_")
        ):
            unit = "mm" if not metric.startswith("mse") else "mm²"
            return metric.upper(), "lower", unit
        if metric in {"r2", "explained_variance"} or metric.startswith(("r2_", "explained_variance_")):
            return metric.upper(), "higher", ""
        if metric in {"fp", "fn"}:
            return metric.upper(), "lower", "count"
        if metric in {"tp", "tn", "n_samples", "samples_visualized", "feature_dim"}:
            return metric.replace("_", " ").title(), "informational", "count"
        if metric == "davies_bouldin":
            return "Davies-Bouldin", "lower", ""
        if metric in {"silhouette", "calinski_harabasz", "pca_explained_variance_2d"}:
            return metric.replace("_", " ").title(), "higher", ""
        if scope.startswith("classification"):
            unit = "%" if metric.endswith("_percent") or "accuracy" in metric or "f1" in metric else ""
            return metric.replace("clf_", "").replace("_", " ").title(), "higher", unit
        return metric.replace("_", " ").title(), "informational", ""

    def add_metric_rows(meta, scope, entity_type, entity_name, values, allowed=None):
        for metric, value in values.items():
            if allowed is not None and metric not in allowed:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_value):
                continue
            metric_label, better_direction, unit = metric_metadata(scope, metric)
            rows.append({
                **meta,
                "scope": scope,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "metric": metric,
                "metric_label": metric_label,
                "value": numeric_value,
                "unit": unit,
                "better_direction": better_direction,
            })

    classification_overall_metrics = {
        "clf_accuracy", "clf_balanced_accuracy",
        "clf_precision_macro", "clf_recall_macro", "clf_f1_macro",
        "clf_precision_micro", "clf_recall_micro", "clf_f1_micro",
        "clf_precision_weighted", "clf_recall_weighted", "clf_f1_weighted",
        "clf_mcc", "clf_cohen_kappa",
    }
    classification_class_metrics = {
        "class_accuracy", "class_accuracy_percent", "precision", "recall",
        "f1", "specificity", "balanced_accuracy_ovr", "mcc_ovr",
        "one_vs_rest_accuracy", "tp", "fp", "fn", "tn", "n_samples",
    }
    regression_metrics = {
        "mae", "mse", "rmse", "r2",
        "median_ae", "max_error", "explained_variance", "mean_error",
        "nrmse", "nrmse_range", "error_mean", "error_std",
        "true_mean", "pred_mean", "n_samples",
    }
    regression_shape_metrics = {
        f"{metric}_{suffix}"
        for metric in ("mae", "rmse", "r2", "max_error", "nrmse", "bias")
        for suffix in ("w", "l", "d", "avg")
    } | {"clf_acc", "n_samples"}
    regression_overall_metrics = {
        f"{metric}_avg"
        for metric in (
            "mae", "mse", "rmse", "r2",
            "median_ae", "max_error", "explained_variance", "mean_error",
            "nrmse", "nrmse_range",
        )
    }

    for item in series_items:
        for pct in item.get("percent_list", PERCENT_LIST):
            for seed in item.get("seed_list", SEEDS):
                seed_tag = f"{item['tag']}_seed_{seed}"
                run_dir = find_latest_run_dir(
                    get_percent_dir(item["mode"], pct),
                    seed_tag,
                )
                if not run_dir:
                    continue
                dedupe_key = os.path.normpath(run_dir)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                alpha, warmup = get_series_alpha_warmup(item)
                mode = item.get("mode", "")
                alpha_value = 0.0 if mode == "no_pinn" else alpha
                warmup_value = 0 if mode == "no_pinn" else warmup
                alpha_text = "0" if alpha_value is None else alpha_to_label(alpha_value)
                warmup_text = "0" if warmup_value is None else str(warmup_value)
                model_id = (
                    f"{mode}_alpha_{alpha_text}_warmup_{warmup_text}"
                    f"_train_{int(pct):02d}pct_seed_{seed}"
                )
                meta = {
                    "model_id": model_id,
                    "model_name": item.get("label", ""),
                    "series_key": item.get("key", ""),
                    "series_label": item.get("label", ""),
                    "group": item.get("group", ""),
                    "mode": mode,
                    "train_percent": int(pct),
                    "seed": str(seed),
                    "alpha": alpha_value,
                    "warmup": warmup_value,
                }

                overall_path = os.path.join(
                    run_dir,
                    "test_classification_metrics_overall.csv",
                )
                if os.path.isfile(overall_path):
                    df = pd.read_csv(overall_path)
                    if not df.empty:
                        add_metric_rows(
                            meta,
                            "classification_overall",
                            "model",
                            "overall",
                            df.iloc[0].to_dict(),
                            classification_overall_metrics,
                        )

                target_path = os.path.join(
                    run_dir,
                    "test_regression_metrics_by_target.csv",
                )
                target_df = (
                    pd.read_csv(target_path)
                    if os.path.isfile(target_path)
                    else pd.DataFrame()
                )

                overall_values = {}
                training_results_path = os.path.join(run_dir, "training_results.csv")
                if os.path.isfile(training_results_path):
                    df = pd.read_csv(training_results_path)
                    if not df.empty:
                        overall_values.update(df.iloc[0].to_dict())

                # Older result files may contain only per-target W/L/D metrics.
                # Fill any missing overall values directly from those metrics.
                if not target_df.empty:
                    for metric in (
                        "mae", "mse", "rmse", "r2", "max_error", "nrmse",
                        "median_ae", "explained_variance",
                    ):
                        column = f"{metric}_avg"
                        current = pd.to_numeric(
                            pd.Series([overall_values.get(column)]),
                            errors="coerce",
                        ).iloc[0]
                        if pd.isna(current) and metric in target_df.columns:
                            values = pd.to_numeric(target_df[metric], errors="coerce")
                            if values.notna().any():
                                overall_values[column] = float(values.mean())

                    current_mean_error = pd.to_numeric(
                        pd.Series([overall_values.get("mean_error_avg")]),
                        errors="coerce",
                    ).iloc[0]
                    if pd.isna(current_mean_error):
                        mean_error_column = next(
                            (
                                column for column in ("mean_error", "error_mean")
                                if column in target_df.columns
                            ),
                            None,
                        )
                        if mean_error_column is not None:
                            values = pd.to_numeric(
                                target_df[mean_error_column],
                                errors="coerce",
                            )
                            if values.notna().any():
                                overall_values["mean_error_avg"] = float(values.mean())

                add_metric_rows(
                    meta,
                    "regression_overall",
                    "model",
                    "average_wld",
                    overall_values,
                    regression_overall_metrics,
                )

                class_path = os.path.join(run_dir, "test_class_metrics.csv")
                if os.path.isfile(class_path):
                    df = pd.read_csv(class_path)
                    for _, row in df.iterrows():
                        add_metric_rows(
                            meta,
                            "classification_per_class",
                            "class",
                            str(row.get("shape", "")),
                            row.to_dict(),
                            classification_class_metrics,
                        )

                if not target_df.empty:
                    for _, row in target_df.iterrows():
                        add_metric_rows(
                            meta,
                            "regression_per_target",
                            "target",
                            str(row.get("target", "")),
                            row.to_dict(),
                            regression_metrics,
                        )

                shape_path = os.path.join(run_dir, "test_metrics_per_shape.csv")
                if os.path.isfile(shape_path):
                    df = pd.read_csv(shape_path)
                    for _, row in df.iterrows():
                        add_metric_rows(
                            meta,
                            "regression_per_class",
                            "class",
                            str(row.get("shape", "")),
                            row.to_dict(),
                            regression_shape_metrics,
                        )

                feature_metrics_path = os.path.join(
                    report_dir,
                    "feature_visualization_by_model",
                    _safe_filename(item.get("key", item.get("label", "model"))),
                    f"train_{int(pct):02d}pct_seed_{seed}",
                    "feature_separation_metrics.csv",
                )
                if not os.path.isfile(feature_metrics_path):
                    embedding_candidates = [
                        os.path.join(run_dir, "test_feature_embeddings.npz"),
                        os.path.join(
                            run_dir,
                            "checkpoint_final",
                            "test_feature_embeddings.npz",
                        ),
                    ]
                    embedding_path = next(
                        (
                            path for path in embedding_candidates
                            if os.path.isfile(path)
                        ),
                        None,
                    )
                    if embedding_path is not None:
                        try:
                            with np.load(embedding_path, allow_pickle=False) as data:
                                feature_output_dir = os.path.dirname(
                                    feature_metrics_path
                                )
                                save_single_model_feature_visualizations(
                                    backbone_features=data["backbone_features"],
                                    regression_features=data["regression_features"],
                                    class_ids=data["true_class"],
                                    true_wld=data["true_wld"],
                                    class_names=data["class_names"].astype(str),
                                    output_dir=feature_output_dir,
                                    title_prefix=(
                                        f"{item.get('label', item.get('key', 'model'))} | "
                                        f"train={pct}% | seed={seed}"
                                    ),
                                )
                            print(
                                "[FEATURE][RECOVER] Rebuilt feature metrics from "
                                f"saved embeddings: {feature_metrics_path}"
                            )
                        except Exception as exc:
                            print(
                                "[FEATURE][WARN] Could not rebuild feature metrics "
                                f"for {item.get('key')} pct={pct} seed={seed}: {exc}"
                            )
                if os.path.isfile(feature_metrics_path):
                    df = pd.read_csv(feature_metrics_path)
                    for _, row in df.iterrows():
                        add_metric_rows(
                            meta,
                            "feature_space",
                            "feature_space",
                            str(row.get("feature_space", "")),
                            row.to_dict(),
                            {
                                "feature_dim", "samples_visualized", "silhouette",
                                "calinski_harabasz", "davies_bouldin",
                                "pca_explained_variance_2d",
                            },
                        )

    output_path = os.path.join(report_dir, "all_metrics_long.csv")
    metrics_long_df = pd.DataFrame(rows)
    _atomic_to_csv(metrics_long_df, output_path, index=False)
    _atomic_to_csv(
        metrics_long_df,
        os.path.join(REPORT_ROOT, "latest_all_metrics_long.csv"),
        index=False,
    )
    if not metrics_long_df.empty:
        metrics_wide_df = metrics_long_df.pivot_table(
            index=[
                "model_id", "model_name", "mode", "alpha", "warmup",
                "train_percent", "seed",
            ],
            columns=["scope", "entity_name", "metric"],
            values="value",
            aggfunc="first",
        )
        metrics_wide_df.columns = [
            "__".join(str(part) for part in col if str(part))
            for col in metrics_wide_df.columns
        ]
        metrics_wide_df = metrics_wide_df.reset_index()
    else:
        metrics_wide_df = pd.DataFrame()
    _atomic_to_csv(
        metrics_wide_df,
        os.path.join(report_dir, "all_metrics_by_model_wide.csv"),
        index=False,
    )
    _atomic_to_csv(
        metrics_wide_df,
        os.path.join(report_dir, "all_metrics.csv"),
        index=False,
    )
    save_grouped_metric_csvs(metrics_wide_df, report_dir)
    _atomic_to_csv(
        metrics_wide_df,
        os.path.join(REPORT_ROOT, "latest_all_metrics_by_model_wide.csv"),
        index=False,
    )
    save_full_metric_tables(metrics_long_df, report_dir)
    print(f"[OK] Saved all model metrics: {output_path} ({len(rows)} rows)")
    return output_path


def save_grouped_metric_csvs(metrics_wide_df, report_dir):
    """Split the wide metric table into independently usable CSV files."""
    if metrics_wide_df.empty:
        return []

    output_dir = os.path.join(report_dir, "metrics_by_group")
    os.makedirs(output_dir, exist_ok=True)
    model_columns = [
        column for column in (
            "model_id", "model_name", "mode", "alpha", "warmup",
            "train_percent", "seed",
        )
        if column in metrics_wide_df.columns
    ]
    group_specs = [
        ("model_information.csv", None),
        ("classification_overall.csv", "classification_overall__"),
        ("classification_per_class.csv", "classification_per_class__"),
        ("regression_overall.csv", "regression_overall__"),
        ("regression_per_target_wld.csv", "regression_per_target__"),
        ("regression_per_shape.csv", "regression_per_class__"),
        ("feature_space.csv", "feature_space__"),
    ]

    saved_paths = []
    for filename, prefix in group_specs:
        if prefix is None:
            columns = model_columns
        else:
            metric_columns = [
                column for column in metrics_wide_df.columns
                if column.startswith(prefix)
            ]
            columns = model_columns + metric_columns
        output_path = os.path.join(output_dir, filename)
        _atomic_to_csv(metrics_wide_df[columns], output_path, index=False)
        saved_paths.append(output_path)

    saved_paths.extend(
        save_per_class_and_target_metric_csvs(
            metrics_wide_df, output_dir, model_columns
        )
    )
    print(f"[OK] Saved grouped metric CSV files: {output_dir}")
    return saved_paths


def save_per_class_and_target_metric_csvs(metrics_wide_df, output_dir, model_columns):
    """Save one CSV per class, per W/L/D target, and per class-target pair."""
    saved_paths = []

    def collect_entities(prefix):
        entities = {}
        for column in metrics_wide_df.columns:
            if not column.startswith(prefix):
                continue
            remainder = column[len(prefix):]
            if "__" not in remainder:
                continue
            entity, metric = remainder.split("__", 1)
            if entity and metric:
                entities.setdefault(entity, []).append((column, metric))
        return entities

    def save_entity_files(directory_name, prefix):
        directory = os.path.join(output_dir, directory_name)
        os.makedirs(directory, exist_ok=True)
        for entity, columns_and_metrics in sorted(collect_entities(prefix).items()):
            source_columns = model_columns + [item[0] for item in columns_and_metrics]
            entity_df = metrics_wide_df[source_columns].copy()
            entity_df.rename(
                columns={column: metric for column, metric in columns_and_metrics},
                inplace=True,
            )
            entity_df.insert(len(model_columns), "entity", entity)
            output_path = os.path.join(directory, f"{_safe_filename(entity)}.csv")
            _atomic_to_csv(entity_df, output_path, index=False)
            saved_paths.append(output_path)

    save_entity_files("classification_by_class", "classification_per_class__")
    save_entity_files("regression_by_target", "regression_per_target__")

    class_entities = collect_entities("regression_per_class__")
    class_target_root = os.path.join(output_dir, "regression_by_class_and_target")
    os.makedirs(class_target_root, exist_ok=True)
    for class_name, columns_and_metrics in sorted(class_entities.items()):
        class_dir = os.path.join(class_target_root, _safe_filename(class_name))
        os.makedirs(class_dir, exist_ok=True)
        for target, suffix in (("W", "_w"), ("L", "_l"), ("D", "_d")):
            selected = [
                (column, metric[:-len(suffix)])
                for column, metric in columns_and_metrics
                if metric.endswith(suffix)
            ]
            shared = [
                (column, metric)
                for column, metric in columns_and_metrics
                if metric in {"n_samples", "clf_acc"}
            ]
            if not selected:
                continue
            source_columns = model_columns + [item[0] for item in selected + shared]
            target_df = metrics_wide_df[source_columns].copy()
            target_df.rename(
                columns={column: metric for column, metric in selected + shared},
                inplace=True,
            )
            target_df.insert(len(model_columns), "class", class_name)
            target_df.insert(len(model_columns) + 1, "target", target)
            output_path = os.path.join(class_dir, f"{target}.csv")
            _atomic_to_csv(target_df, output_path, index=False)
            saved_paths.append(output_path)

    print(
        f"[OK] Saved per-class and per-target metric CSV files: {output_dir}"
    )
    return saved_paths


def _safe_filename(text, max_len=180):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return (safe or "run")[:max_len]


def _prepare_experiment_jobs(script_name, env_vars, name, custom_tag, progress_callback=None):
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {name}")
    print(f"{'='*60}")
    mode = env_vars.get("_MODE", "pinn")
    percent_list = list(env_vars.get("_PERCENT_LIST", PERCENT_LIST))
    seed_list = [str(s) for s in env_vars.get("_SEEDS", SEEDS)]
    jobs = []

    for pct in percent_list:
        print(f"\n---> Training {pct}% ...")
        env = os.environ.copy()
        env_clean = {k: v for k, v in env_vars.items() if not k.startswith("_")}
        env.update(env_clean)
        env["TRAIN_PERCENT"] = str(pct)
        env["BATCH_SIZE"] = str(BATCH_SIZE)

        pct_completed = 0

        # Run across seeds; tag runs per-seed so they are stored separately and discoverable.
        for seed in seed_list:
            seed_tag = f"{custom_tag}_seed_{seed}"
            run_unit = (mode, pct, str(seed), seed_tag)

            with RUN_UNITS_LOCK:
                already_handled = run_unit in RUN_UNITS_DONE
            if already_handled:
                print(f"[SKIP-DUP] seed={seed} pct={pct} tag={custom_tag} already handled in this run")
                pct_completed += 1
                continue

            reuse_completed = SKIP_IF_EXISTS or (
                mode == "no_pinn" and REUSE_COMPLETED_NO_PINN
            )
            if reuse_completed and has_completed_seed_result(mode, pct, seed_tag):
                print(
                    f"[SKIP-TRAIN] seed={seed} pct={pct} already has a final model; "
                    "it will be re-inferred and re-plotted"
                )
                with RUN_UNITS_LOCK:
                    RUN_UNITS_DONE.add(run_unit)
                pct_completed += 1
                continue

            env_seed = env.copy()
            env_seed["CUSTOM_RUN_TAG"] = seed_tag
            env_seed["RANDOM_STATE"] = str(seed)
            env_seed["PYTHONPATH"] = SCRIPT_DIR + os.pathsep + env_seed.get("PYTHONPATH", "")
            if DATA_PATH:
                env_seed["DATA_PATH"] = DATA_PATH
            if LABELS_PATH:
                env_seed["LABELS_PATH"] = LABELS_PATH

            # Create experiment directory
            if EXP_MANAGER:
                config_dict = {
                    'mode': mode,
                    'percent': pct,
                    'seed': int(seed),
                    'alpha': float(env_seed.get('ALPHA_INIT', 0.1)),
                    'warmup_epoch': int(env_seed.get('PINN_ACTIVATION_EPOCH', 100)),
                    'batch_size': BATCH_SIZE,
                    'epochs': EPOCHS,
                    'alpha_values': ALPHA_VALUES,
                }
                exp_dir = EXP_MANAGER.create_exp_dir(mode, pct, int(seed), config_dict)
                EXP_MANAGER.log_run(exp_dir, f"Starting training for {mode} p={pct}% seed={seed}")

            # AUTO_RESUME: try to find a resume checkpoint for this seed/pct
            if AUTO_RESUME:
                resume_ckpt = find_latest_checkpoint_for_percent(mode, pct, seed_tag)
                if resume_ckpt == "__COMPLETED__":
                    print(f"[SKIP] seed={seed} pct={pct} already completed (found final checkpoint)")
                    with RUN_UNITS_LOCK:
                        RUN_UNITS_DONE.add(run_unit)
                    pct_completed += 1
                    if progress_callback is not None:
                        try:
                            progress_callback(reason=f"skip-final seed={seed} pct={pct} tag={custom_tag}")
                        except Exception as exc:
                            print(f"[LIVE][WARN] Progress callback failed after skip: {exc}")
                    continue
                if resume_ckpt:
                    env_seed["RESUME_CHECKPOINT"] = resume_ckpt
                    print(f"[RESUME] seed={seed} pct={pct} from: {resume_ckpt}")
                else:
                    env_seed["RESUME_CHECKPOINT"] = ""
            else:
                # A stale shell variable must not make a clean retraining run
                # resume one of the invalid pre-gradient-fix checkpoints.
                env_seed["RESUME_CHECKPOINT"] = ""

            with RUN_UNITS_LOCK:
                if run_unit in RUN_UNITS_DONE:
                    print(f"[SKIP-DUP] seed={seed} pct={pct} tag={custom_tag} already queued in this run")
                    pct_completed += 1
                    continue
                # Mark as queued so equivalent deduped series are not submitted twice.
                RUN_UNITS_DONE.add(run_unit)

            jobs.append(
                {
                    "script_name": script_name,
                    "env_seed": env_seed,
                    "name": name,
                    "custom_tag": custom_tag,
                    "seed_tag": seed_tag,
                    "run_unit": run_unit,
                    "mode": mode,
                    "pct": pct,
                    "seed": str(seed),
                    "exp_dir": exp_dir if EXP_MANAGER else None,
                    "display": f"{name} | pct={pct}% seed={seed}",
                }
            )

        if pct_completed == len(seed_list):
            print(f"[INFO] All seeds already completed for {pct}% in {custom_tag}")

    return jobs


def _run_training_job(job):
    cmd = [PYTHON_CMD, job["script_name"]]
    start = time.time()
    log_path = job.get("log_path")
    try:
        if job.get("use_log"):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write(f"Workdir: {SCRIPT_DIR}\n")
                log_f.write(f"CUDA_VISIBLE_DEVICES: {job['env_seed'].get('CUDA_VISIBLE_DEVICES', '')}\n")
                log_f.write("=" * 80 + "\n")
                log_f.flush()
                subprocess.run(
                    cmd,
                    env=job["env_seed"],
                    cwd=SCRIPT_DIR,
                    check=True,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
        else:
            subprocess.run(cmd, env=job["env_seed"], cwd=SCRIPT_DIR, check=True)
        return {
            "ok": True,
            "job": job,
            "elapsed": time.time() - start,
        }
    except subprocess.CalledProcessError as exc:
        return {
            "ok": False,
            "job": job,
            "elapsed": time.time() - start,
            "error": str(exc),
            "returncode": exc.returncode,
        }
    except Exception as exc:
        return {
            "ok": False,
            "job": job,
            "elapsed": time.time() - start,
            "error": repr(exc),
            "returncode": None,
        }


def _run_prepared_jobs(jobs, progress_callback=None):
    if not jobs:
        print("[INFO] No training jobs to run in this phase.")
        return

    worker_count = min(PARALLEL_RUNS, len(jobs))
    use_logs = worker_count > 1
    log_dir = os.path.join(REPORT_DIR, "run_logs")

    print(f"\n[PARALLEL] Queued jobs: {len(jobs)} | workers: {worker_count}")
    if use_logs:
        print(f"[PARALLEL] Subprocess output will be written to: {log_dir}")

    for idx, job in enumerate(jobs):
        _configure_job_runtime(job, idx, use_logs, log_dir)

    if worker_count == 1:
        for job in jobs:
            result = _run_training_job(job)
            _handle_job_result(result, progress_callback)
        return

    failed_result = None
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_run_training_job, job) for job in jobs]
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                result = future.result()
            except CancelledError:
                continue
            if not result["ok"] and failed_result is None:
                failed_result = result
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
            _handle_job_result(result, progress_callback, exit_on_failure=False)

    if failed_result is not None:
        job = failed_result["job"]
        print(f"[ERROR] At least one training job failed: {job['display']}")
        sys.exit(1)


def _handle_job_result(result, progress_callback=None, exit_on_failure=True):
    job = result["job"]
    elapsed_min = result["elapsed"] / 60.0
    if result["ok"]:
        print(f"[DONE] {job['display']} finished in {elapsed_min:.1f} min")
        if EXP_MANAGER and job.get("exp_dir"):
            EXP_MANAGER.log_run(job["exp_dir"], "Training completed successfully")
        if progress_callback is not None:
            try:
                progress_callback(reason=f"done seed={job['seed']} pct={job['pct']} tag={job['custom_tag']}")
            except Exception as exc:
                print(f"[LIVE][WARN] Progress callback failed after completion: {exc}")
        return

    log_hint = f" | log={job.get('log_path')}" if job.get("log_path") else ""
    print(f"[ERROR] Failed training {job['pct']}% for {job['name']} seed={job['seed']}: {result['error']}{log_hint}")
    if EXP_MANAGER and job.get("exp_dir"):
        EXP_MANAGER.log_run(job["exp_dir"], f"Training FAILED: {result['error']}")
    if exit_on_failure:
        sys.exit(1)


def _configure_job_runtime(job, job_index, use_logs, log_dir):
    gpu_id = GPU_IDS[job_index % len(GPU_IDS)]
    job["gpu_id"] = gpu_id
    job["env_seed"]["CUDA_VISIBLE_DEVICES"] = gpu_id
    job["use_log"] = use_logs
    if use_logs:
        log_name = _safe_filename(
            f"{job['mode']}_pct{job['pct']:02d}_seed{job['seed']}_{job['seed_tag']}.log"
        )
        job["log_path"] = os.path.join(log_dir, log_name)
    print(
        f"[QUEUE] {job['display']} | gpu={gpu_id}"
        + (f" | log={job['log_path']}" if use_logs else "")
    )


def _interleave_job_groups(*job_groups):
    """Round-robin job groups so one large group cannot starve another."""
    queues = [list(group) for group in job_groups if group]
    jobs = []
    while queues:
        next_queues = []
        for queue in queues:
            jobs.append(queue.pop(0))
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return jobs


def run_experiment(script_name, env_vars, name, custom_tag, progress_callback=None):
    jobs = _prepare_experiment_jobs(script_name, env_vars, name, custom_tag, progress_callback)
    _run_prepared_jobs(jobs, progress_callback)


def _prepare_experiment_group_jobs(series_items, start_index, progress_callback=None):
    jobs = []
    for offset, item in enumerate(series_items, start=start_index):
        env = dict(item["env"])
        env["_PERCENT_LIST"] = list(item.get("percent_list", PERCENT_LIST))
        env["_SEEDS"] = list(item.get("seed_list", SEEDS))
        jobs.extend(
            _prepare_experiment_jobs(
                item["script"],
                env,
                f"{offset}. {item['label']}",
                item["tag"],
                progress_callback=progress_callback,
            )
        )
    return jobs


def _prepare_grouped_experiment_jobs(series_items, start_index, progress_callback=None):
    grouped_jobs = []
    for offset, item in enumerate(series_items, start=start_index):
        grouped_jobs.append(
            _prepare_experiment_group_jobs([item], offset, progress_callback)
        )
    return _interleave_job_groups(*grouped_jobs)


def run_experiment_group(series_items, start_index, progress_callback=None):
    jobs = _prepare_grouped_experiment_jobs(series_items, start_index, progress_callback)
    _run_prepared_jobs(jobs, progress_callback)


def run_auto_alpha_pipeline(
    *,
    series,
    baseline_series,
    alpha_series,
    deferred_series,
    alpha_percent_priority,
    update_warmup,
    update_seed,
    progress_callback=None,
):
    """Run alpha-dependent workflow without letting baseline block deferred studies."""
    baseline_jobs = _prepare_experiment_group_jobs(baseline_series, 1, progress_callback)
    alpha_start = len(baseline_series) + 1
    alpha_jobs = _prepare_experiment_group_jobs(alpha_series, alpha_start, progress_callback)
    for job in baseline_jobs:
        job["_phase"] = "baseline"
    for job in alpha_jobs:
        job["_phase"] = "alpha"

    # Start alpha jobs immediately because deferred studies depend on their result.
    # Baseline jobs are still included in the same executor and run alongside them.
    initial_jobs = _interleave_job_groups(alpha_jobs, baseline_jobs)
    pending = set()
    failed_result = None
    deferred_submitted = False
    remaining_alpha_jobs = len(alpha_jobs)
    job_index = 0
    use_logs = PARALLEL_RUNS > 1
    log_dir = os.path.join(REPORT_DIR, "run_logs")

    def _build_deferred_jobs():
        reevaluate_existing_models_for_series(alpha_series)
        collect_series_results(series)
        selected_alpha = choose_best_alpha_from_alpha_series(series, alpha_percent_priority)
        if selected_alpha is not None:
            retarget_deferred_pinn_alpha(
                series,
                selected_alpha,
                update_warmup=update_warmup,
                update_seed=update_seed,
            )
            dedupe_equivalent_series_tags(series)
            apply_distinct_series_styles(series)
            print(
                f"[AUTO] Deferred PINN studies use alpha={alpha_to_label(selected_alpha)} "
                f"(warmup_update={update_warmup}, seed_update={update_seed})"
            )
        else:
            print("[AUTO][WARN] No completed alpha sweep result found; deferred studies keep configured alpha.")

        deferred_now = [s for s in series if s.get("group") in {"warmup", "seed"}]
        deferred_start = len(baseline_series) + len(alpha_series) + 1
        jobs = _prepare_grouped_experiment_jobs(deferred_now, deferred_start, progress_callback)
        for job in jobs:
            job["_phase"] = "deferred"
        return jobs

    if not initial_jobs and not deferred_series:
        print("[INFO] No training jobs to run in this pipeline.")
        return

    print(f"\n[PARALLEL] Pipeline workers: {PARALLEL_RUNS}")
    if use_logs:
        print(f"[PARALLEL] Subprocess output will be written to: {log_dir}")

    with ThreadPoolExecutor(max_workers=PARALLEL_RUNS) as executor:
        def _submit_jobs(jobs):
            nonlocal job_index
            for job in jobs:
                _configure_job_runtime(job, job_index, use_logs, log_dir)
                job_index += 1
                future = executor.submit(_run_training_job, job)
                pending.add(future)

        _submit_jobs(initial_jobs)

        if remaining_alpha_jobs == 0 and not deferred_submitted:
            deferred_submitted = True
            _submit_jobs(_build_deferred_jobs())

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except CancelledError:
                    continue

                job = result["job"]
                if job.get("_phase") == "alpha":
                    remaining_alpha_jobs -= 1

                if not result["ok"] and failed_result is None:
                    failed_result = result
                    for pending_future in pending:
                        pending_future.cancel()

                _handle_job_result(result, progress_callback, exit_on_failure=False)

            if (
                failed_result is None
                and remaining_alpha_jobs == 0
                and not deferred_submitted
            ):
                deferred_submitted = True
                _submit_jobs(_build_deferred_jobs())

    if failed_result is not None:
        job = failed_result["job"]
        print(f"[ERROR] At least one training job failed: {job['display']}")
        sys.exit(1)

def extract_metrics(mode, suffix):
    pct_pattern = re.compile(r"train_(\d+)pct")
    results = {}
    folder_root = OUTPUT_NO_PINN_ROOT if mode == "no_pinn" else OUTPUT_PINN_ROOT
    if not os.path.exists(folder_root):
        return results

    for pct_folder in os.listdir(folder_root):
        match = pct_pattern.match(pct_folder)
        if not match:
            continue
        percent = int(match.group(1))
        
        pct_path = os.path.join(folder_root, pct_folder)
        if not os.path.isdir(pct_path):
            continue
            
        # Find run folders matching the provided suffix prefix (includes seed-tagged runs)
        run_folders = sorted(
            [f for f in os.listdir(pct_path) if f.startswith(suffix + "_")],
            reverse=True,
        )
        if not run_folders:
            # also allow exact match (no trailing underscore)
            run_folders = sorted([f for f in os.listdir(pct_path) if f.startswith(suffix)], reverse=True)
            if not run_folders:
                continue

        for run_folder in run_folders:
            latest_run = os.path.join(pct_path, run_folder)
            if REPORT_REEVAL_ONLY and os.path.normpath(latest_run) not in _REEVAL_SUCCESS_RUN_DIRS:
                continue
            csv_file = os.path.join(latest_run, "training_results.csv")
            if not os.path.exists(csv_file):
                continue
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0 and "clf_acc" in df.columns:
                    row = df.iloc[0]
                    clf_acc_raw = float(df.iloc[0]["clf_acc"])
                    clf_acc_pct = clf_acc_raw * 100.0 if clf_acc_raw <= 1.0 else clf_acc_raw
                    mae_val = float(df.iloc[0]["mae_avg"]) if "mae_avg" in df.columns else (float(df.iloc[0]["mae"]) if "mae" in df.columns else np.nan)
                    rmse_val = float(df.iloc[0]["rmse_avg"]) if "rmse_avg" in df.columns else (float(df.iloc[0]["rmse"]) if "rmse" in df.columns else np.nan)
                    r2_val = float(df.iloc[0]["r2_avg"]) if "r2_avg" in df.columns else (float(df.iloc[0]["r2"]) if "r2" in df.columns else np.nan)
                    nrmse_val = float(df.iloc[0]["nrmse_avg"]) if "nrmse_avg" in df.columns else (float(df.iloc[0]["nrmse_range_avg"]) if "nrmse_range_avg" in df.columns else (float(df.iloc[0]["nrmse"]) if "nrmse" in df.columns else np.nan))
                    max_err_val = float(df.iloc[0]["max_error_avg"]) if "max_error_avg" in df.columns else (float(df.iloc[0]["max_error"]) if "max_error" in df.columns else np.nan)
                    bal_acc_val = float(df.iloc[0]["clf_balanced_accuracy"]) if "clf_balanced_accuracy" in df.columns else (float(df.iloc[0]["clf_balanced_acc"]) if "clf_balanced_acc" in df.columns else np.nan)
                    f1_macro_val = float(df.iloc[0]["clf_f1_macro"]) if "clf_f1_macro" in df.columns else (float(df.iloc[0]["f1_macro"]) if "f1_macro" in df.columns else np.nan)
                    mcc_val = float(df.iloc[0]["clf_mcc"]) if "clf_mcc" in df.columns else (float(df.iloc[0]["mcc"]) if "mcc" in df.columns else np.nan)

                    rec = {
                        "acc": clf_acc_pct,
                        "mae": mae_val,
                        "rmse": rmse_val,
                        "r2": r2_val,
                        "nrmse": nrmse_val,
                        "max_error": max_err_val,
                        "balanced_accuracy": bal_acc_val * 100.0 if np.isfinite(bal_acc_val) and bal_acc_val <= 1.0 else bal_acc_val,
                        "f1_macro": f1_macro_val * 100.0 if np.isfinite(f1_macro_val) and f1_macro_val <= 1.0 else f1_macro_val,
                        "mcc": mcc_val,
                        "run_dir": latest_run,
                        "csv_file": csv_file,
                    }
                    for col, val in row.items():
                        try:
                            rec[col] = float(val)
                        except Exception:
                            continue
                    results[percent] = rec
                    break
            except Exception:
                continue
    return results


def aggregate_seed_results(per_seed_results):
    """Average numeric metrics across seeds for each train percent."""
    by_pct = defaultdict(list)
    for seed, seed_results in per_seed_results.items():
        for pct, rec in seed_results.items():
            by_pct[pct].append(rec)

    aggregated = {}
    for pct, records in by_pct.items():
        out = {
            "run_dir": records[0].get("run_dir", ""),
            "csv_file": records[0].get("csv_file", ""),
            "seed_count": len(records),
        }
        numeric_keys = sorted({
            key
            for rec in records
            for key, val in rec.items()
            if key not in {"run_dir", "csv_file"}
        })
        for key in numeric_keys:
            vals = []
            for rec in records:
                try:
                    vals.append(float(rec[key]))
                except Exception:
                    continue
            out[key] = float(np.mean(vals)) if vals else np.nan
        aggregated[pct] = out
    return aggregated


def consolidate_loss_logs(tgt_pcts, series_dict, report_dir):
    """
    Gộp tất cả epoch_loss_log.csv từ từng run thành một file CSV tổng hợp.
    Schema chung: epoch, pinn_loss, data_loss, total_loss,
                  train_percent, series_label, mode, alpha
    """
    loss_root = os.path.join(report_dir, "loss_logs")
    os.makedirs(loss_root, exist_ok=True)

    frames = []
    for pct in tgt_pcts:
        for item in series_dict:
            rec = item.get("results", {}).get(pct)
            if rec is None:
                continue
            log_path = os.path.join(rec["run_dir"], "epoch_loss_log.csv")
            if not os.path.isfile(log_path):
                print(f"[WARN] Không tìm thấy epoch_loss_log.csv: {log_path}")
                continue
            try:
                df = pd.read_csv(log_path)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"[WARN] Lỗi đọc {log_path}: {exc}")

    output_csv = os.path.join(loss_root, "all_runs_epoch_loss.csv")
    if frames:
        _atomic_to_csv(pd.concat(frames, ignore_index=True), output_csv, index=False)
        total_rows = sum(len(f) for f in frames)
        print(f"[OK] Saved consolidated epoch loss log: {output_csv} ({total_rows} rows)")
    else:
        print("[INFO] No epoch_loss_log.csv files found to consolidate.")

    return output_csv


def save_line_plot(tgt_pcts, series_dict, metric_key, title, y_label, out_file, lower_is_better):
    """Publication-quality line plot comparison across training percentages."""
    import matplotlib.patheffects as path_effects

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
    plotted_any = False

    plot_series = collapse_equivalent_plot_series(series_dict)

    palette_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    palette_markers = ["o", "s", "^", "D", "v", "P"]

    for s_idx, item in enumerate(plot_series):
        is_nopinn = item.get("mode") == "no_pinn"
        if len(plot_series) <= 2:
            label = "No-PINN (Baseline)" if is_nopinn else "PINN (Ours)"
            color = "#1f77b4" if is_nopinn else "#d62728"
            linestyle = "--" if is_nopinn else "-"
            marker = "o" if is_nopinn else "s"
        else:
            label = item.get("short_label", item.get("label", f"Series {s_idx+1}"))
            color = item.get("color", palette_colors[s_idx % len(palette_colors)])
            linestyle = item.get("linestyle", "--" if is_nopinn else "-")
            marker = item.get("marker", palette_markers[s_idx % len(palette_markers)])

        values = []
        for p in tgt_pcts:
            info = item["results"].get(p, {})
            val = info.get(metric_key, np.nan)
            if not np.isfinite(val):
                if metric_key == "mae":
                    val = info.get("mae_avg", np.nan)
                elif metric_key == "rmse":
                    val = info.get("rmse_avg", np.nan)
                elif metric_key == "r2":
                    val = info.get("r2_avg", np.nan)
                elif metric_key in ("nrmse", "nrmse_range"):
                    val = info.get("nrmse_avg", info.get("nrmse_range_avg", np.nan))
                elif metric_key == "max_error":
                    val = info.get("max_error_avg", np.nan)
                elif metric_key == "acc":
                    val = info.get("clf_acc_percent", info.get("clf_acc", np.nan))
                    if np.isfinite(val) and val <= 1.0:
                        val = val * 100.0
                elif metric_key in ("balanced_accuracy", "clf_balanced_accuracy", "balanced_acc"):
                    val = info.get("clf_balanced_accuracy", info.get("clf_balanced_acc", info.get("balanced_accuracy", np.nan)))
                    if np.isfinite(val) and val <= 1.0:
                        val = val * 100.0
                elif metric_key in ("f1_macro", "clf_f1_macro"):
                    val = info.get("clf_f1_macro", info.get("f1_macro", np.nan))
                    if np.isfinite(val) and val <= 1.0:
                        val = val * 100.0
                elif metric_key in ("mcc", "clf_mcc"):
                    val = info.get("clf_mcc", info.get("mcc", np.nan))
            elif metric_key in ("acc", "balanced_accuracy", "clf_balanced_accuracy", "balanced_acc", "f1_macro", "clf_f1_macro") and np.isfinite(val) and val <= 1.0:
                val = val * 100.0
            values.append(val)

        try:
            finite_values = np.asarray(values, dtype=float)
        except Exception:
            finite_values = np.asarray([np.nan for _ in values], dtype=float)
        if not np.isfinite(finite_values).any():
            continue

        valid_pts = [(p, v) for p, v in zip(tgt_pcts, values) if np.isfinite(v)]
        if not valid_pts:
            continue

        plot_x, plot_y = zip(*valid_pts)
        ax.plot(
            plot_x,
            plot_y,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.6,
            markersize=9,
            markeredgewidth=1.5,
            markeredgecolor="white",
            color=color,
            label=label,
            alpha=0.92,
            zorder=4,
        )

        if len(plot_series) > 2:
            offsets = [-20, 16, 28, -32, 40, -42]
            y_offset = offsets[s_idx % len(offsets)]
            va = "top" if y_offset < 0 else "bottom"
        else:
            y_offset = -16 if is_nopinn else 14
            va = "top" if is_nopinn else "bottom"

        for p, v in valid_pts:
            if metric_key in ("acc", "balanced_acc", "f1_macro", "nrmse") or "%" in y_label:
                val_text = f"{v:.2f}%"
            else:
                val_text = f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
            txt = ax.annotate(
                val_text,
                (p, v),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center",
                va=va,
                fontsize=11.5,
                fontweight="bold",
                color=color,
                zorder=5,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground="white")])
        plotted_any = True

    ax.set_xlabel("Training Data Percentage (%)", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_ylabel(y_label, fontsize=15, fontweight="bold", labelpad=8)
    ax.set_xticks(tgt_pcts)
    ax.set_xticklabels([f"{p}%" for p in tgt_pcts], fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=13, width=1.2, length=5)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=1)
    ax.margins(x=0.05, y=0.15)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")

    if plotted_any:
        legend = ax.legend(
            fontsize=13,
            loc="upper right" if lower_is_better else "lower right",
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            fancybox=True,
            shadow=False,
        )
        legend.get_frame().set_linewidth(1.0)
    else:
        ax.text(
            0.5,
            0.5,
            f"No completed data for {metric_key}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="dimgray",
        )

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    direction = "Lower is better" if lower_is_better else "Higher is better"
    print(f"[OK] Saved publication plot: {out_file} ({direction})")


def save_percent_bar_plots(tgt_pcts, series_dict, report_dir):
    for pct in tgt_pcts:
        percent_dir = os.path.join(report_dir, "by_percent", f"train_{pct:02d}pct")
        os.makedirs(percent_dir, exist_ok=True)

        labels = []
        colors = []
        mae_vals = []
        rmse_vals = []
        r2_vals = []
        nrmse_vals = []
        acc_vals = []
        f1_vals = []
        mcc_vals = []
        rows = []

        for item in series_dict:
            res = item["results"].get(pct)
            if res is None:
                continue
            labels.append("nopinn" if item.get("mode") == "no_pinn" else "pinn")
            colors.append(item["color"])
            
            mae_vals.append(float(res.get("mae", res.get("mae_avg", np.nan))))
            rmse_vals.append(float(res.get("rmse", res.get("rmse_avg", np.nan))))
            r2_vals.append(float(res.get("r2", res.get("r2_avg", np.nan))))
            nrmse_vals.append(float(res.get("nrmse", res.get("nrmse_avg", np.nan))))
            
            acc = float(res.get("acc", res.get("clf_acc_percent", res.get("clf_acc", np.nan))))
            if np.isfinite(acc) and acc <= 1.0:
                acc = acc * 100.0
            acc_vals.append(acc)
            
            f1 = float(res.get("f1_macro", res.get("clf_f1_macro", np.nan)))
            if np.isfinite(f1) and f1 <= 1.0:
                f1 = f1 * 100.0
            f1_vals.append(f1)
            
            mcc_vals.append(float(res.get("mcc", res.get("clf_mcc", np.nan))))
            
            row = {
                "train_percent": pct,
                "series": item["label"],
                "mae_avg": mae_vals[-1],
                "rmse_avg": rmse_vals[-1],
                "r2_avg": r2_vals[-1],
                "nrmse_avg": nrmse_vals[-1],
                "clf_acc_percent": acc_vals[-1],
                "clf_f1_macro": f1_vals[-1],
                "clf_mcc": mcc_vals[-1],
                "run_dir": res.get("run_dir", ""),
                "csv_file": res.get("csv_file", ""),
            }
            for key, value in res.items():
                if key in {"acc", "balanced_acc", "f1_macro", "mcc", "run_dir", "csv_file"}:
                    continue
                row[key] = value
            rows.append(row)

        _atomic_to_csv(pd.DataFrame(rows), os.path.join(percent_dir, "metrics_table.csv"), index=False)

        x = np.arange(len(labels))

        bar_configs = [
            ("rmse_bar.png", rmse_vals, "Average RMSE (mm) -> Lower is better"),
            ("mae_bar.png", mae_vals, "Average MAE (mm) -> Lower is better"),
            ("nrmse_bar.png", nrmse_vals, "Average NRMSE (%) -> Lower is better"),
            ("r2_bar.png", r2_vals, "Average R² Score -> Higher is better"),
            ("acc_bar.png", acc_vals, "Classification Accuracy (%) -> Higher is better"),
            ("f1_macro_bar.png", f1_vals, "Macro F1-Score (%) -> Higher is better"),
            ("mcc_bar.png", mcc_vals, "Matthews Correlation Coefficient (MCC) -> Higher is better"),
        ]

        for fname, vals, ylabel in bar_configs:
            valid_vals = np.asarray(vals, dtype=float)
            if not np.isfinite(valid_vals).any():
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            try:
                bars = ax.bar(x, valid_vals, color=colors)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold")
                ax.tick_params(axis="y", labelsize=16)
                ax.set_title("")
                ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")
                ax.bar_label(bars, fmt="%.4g", fontsize=16, fontweight="bold")
                ax.grid(axis="y", linestyle="--", alpha=0.45)
                fig.tight_layout()
                out_path = os.path.join(percent_dir, fname)
                fig.savefig(out_path, dpi=220, bbox_inches="tight")
            except Exception as e:
                print(f"[WARN] Failed saving percent bar {fname}: {e}")
            finally:
                plt.close(fig)

    print(f"[OK] Saved per-percent bar charts to: {os.path.join(report_dir, 'by_percent')}")


def save_warmup_feature_visualizations(completed, records, out_dir, percent):
    """Visualize how warmup changes shared and regression feature spaces."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler

    loaded = []
    sample_indices_by_size = {}
    for item, record in zip(completed, records):
        feature_path = os.path.join(
            record.get("run_dir", ""),
            "test_feature_embeddings.npz",
        )
        if not os.path.isfile(feature_path):
            continue
        try:
            with np.load(feature_path) as data:
                n_samples = len(data["true_class"])
                if n_samples not in sample_indices_by_size:
                    keep = np.arange(n_samples)
                    if n_samples > 750:
                        rng = np.random.default_rng(42)
                        keep = np.sort(rng.choice(n_samples, size=750, replace=False))
                    sample_indices_by_size[n_samples] = keep
                keep = sample_indices_by_size[n_samples]
                loaded.append({
                    "mode": item.get("mode", "pinn"),
                    "warmup": (
                        None if item.get("mode") == "no_pinn"
                        else int(item["warmup"])
                    ),
                    "config_label": (
                        "No PINN" if item.get("mode") == "no_pinn"
                        else f"PINN W={int(item['warmup'])}"
                    ),
                    "config_order": (
                        -1 if item.get("mode") == "no_pinn"
                        else int(item["warmup"])
                    ),
                    "backbone": data["backbone_features"][keep],
                    "regression": data["regression_features"][keep],
                    "class_ids": data["true_class"][keep],
                    "true_wld": data["true_wld"][keep],
                    "class_names": data["class_names"].astype(str).tolist(),
                })
        except Exception as exc:
            print(f"[WARN] Failed loading feature embeddings {feature_path}: {exc}")

    if not loaded:
        print("[INFO] No feature embeddings found for warmup visualization.")
        return

    feature_dir = os.path.join(out_dir, "feature_visualization")
    os.makedirs(feature_dir, exist_ok=True)
    separation_rows = []

    for feature_key, feature_label in [
        ("backbone", "Shared CNN backbone"),
        ("regression", "Regression embedding"),
    ]:
        combined = np.concatenate([entry[feature_key] for entry in loaded], axis=0)
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        pca_2d = PCA(n_components=2, random_state=42)
        combined_pca = pca_2d.fit_transform(combined_scaled)
        pre_tsne_dim = min(50, combined_scaled.shape[1], combined_scaled.shape[0] - 1)
        pre_tsne = PCA(n_components=pre_tsne_dim, random_state=42).fit_transform(combined_scaled)
        perplexity = min(30.0, max(5.0, (len(pre_tsne) - 1) / 3.0))
        tsne_signature = inspect.signature(TSNE)
        learning_rate_default = tsne_signature.parameters["learning_rate"].default
        tsne_kwargs = {
            "n_components": 2,
            "perplexity": perplexity,
            "learning_rate": (
                "auto" if isinstance(learning_rate_default, str) else 200.0
            ),
            "init": "pca",
            "random_state": 42,
            "method": "barnes_hut",
            "n_jobs": -1,
        }
        iteration_arg = (
            "max_iter"
            if "max_iter" in tsne_signature.parameters
            else "n_iter"
        )
        tsne_kwargs[iteration_arg] = 1000
        combined_tsne = TSNE(**tsne_kwargs).fit_transform(pre_tsne)

        projection_sets = []
        for method_name, combined_projection in [
            ("pca", combined_pca),
            ("tsne", combined_tsne),
        ]:
            cursor = 0
            projected = []
            for entry in loaded:
                count = len(entry[feature_key])
                entry_projection = combined_projection[cursor:cursor + count]
                cursor += count
                projected.append((entry, entry_projection))
            projection_sets.append((method_name, projected))

        for entry in loaded:
            scaled_features = scaler.transform(entry[feature_key])
            class_ids = entry["class_ids"]
            if len(np.unique(class_ids)) > 1:
                silhouette = float(
                    silhouette_score(
                        scaled_features,
                        class_ids,
                        sample_size=min(1000, len(class_ids)),
                        random_state=42,
                    )
                )
                calinski = float(calinski_harabasz_score(scaled_features, class_ids))
                davies = float(davies_bouldin_score(scaled_features, class_ids))
            else:
                silhouette = calinski = davies = float("nan")

            centroids = [
                np.mean(scaled_features[class_ids == class_id], axis=0)
                for class_id in sorted(np.unique(class_ids))
            ]
            centroid_distances = [
                float(np.linalg.norm(centroids[i] - centroids[j]))
                for i in range(len(centroids))
                for j in range(i + 1, len(centroids))
            ]
            separation_rows.append({
                "train_percent": percent,
                "alpha": 1,
                "seed": 42,
                "warmup": entry["warmup"],
                "mode": entry["mode"],
                "config_label": entry["config_label"],
                "config_order": entry["config_order"],
                "feature_space": feature_key,
                "feature_dim": int(entry[feature_key].shape[1]),
                "samples_visualized": int(len(class_ids)),
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
                "mean_class_centroid_distance": (
                    float(np.mean(centroid_distances))
                    if centroid_distances else float("nan")
                ),
            })

        pub_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        reg_cmaps = {"W": "viridis", "L": "plasma", "D": "turbo"}
        unit_map = {"W": "mm", "L": "mm", "D": "mm"}

        for method_name, projected in projection_sets:
            method_label = "PCA" if method_name == "pca" else "t-SNE"
            n_panels = len(projected)
            ncols = min(2, n_panels)
            nrows = int(math.ceil(n_panels / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(8.5 * ncols, 6.5 * nrows),
                dpi=300,
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for ax, (entry, projection) in zip(axes.flat, projected):
                for class_id, class_name in enumerate(entry["class_names"]):
                    mask = entry["class_ids"] == class_id
                    c_color = pub_colors[class_id % len(pub_colors)]
                    ax.scatter(
                        projection[mask, 0],
                        projection[mask, 1],
                        s=55,
                        alpha=0.82,
                        color=c_color,
                        edgecolors="black",
                        linewidth=0.3,
                        label=class_name,
                    )
                ax.set_title(entry["config_label"], fontsize=15, fontweight="bold", pad=8)
                ax.set_xlabel(f"Joint {method_label} Dim 1", fontsize=14, fontweight="bold")
                ax.set_ylabel(f"Joint {method_label} Dim 2", fontsize=14, fontweight="bold")
                ax.tick_params(axis="both", which="major", labelsize=12)
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)
            for ax in axes.flat[n_panels:]:
                ax.axis("off")
            handles, legend_labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                legend_labels,
                loc="center right",
                title="Crack Shape",
                title_fontsize=14,
                fontsize=13,
                frameon=True,
                facecolor="white",
                edgecolor="#cccccc",
                framealpha=0.95,
            )
            extra_title = ""
            if method_name == "pca":
                extra_title = (
                    f" (Expl. Var: {pca_2d.explained_variance_ratio_.sum() * 100:.1f}%)"
                )
            fig.suptitle(
                f"{feature_label} — Multi-Model Joint {method_label} by Class (Train {percent}%){extra_title}",
                fontsize=18,
                fontweight="bold",
                y=0.98,
            )
            fig.tight_layout(rect=(0, 0, 0.88, 0.95))
            for ext in (".png", ".pdf"):
                fig.savefig(
                    os.path.join(
                        feature_dir,
                        f"{feature_key}_joint_{method_name}_by_class{ext}",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)

            if feature_key == "regression":
                for target_idx, target_name in enumerate(("W", "L", "D")):
                    target_values = np.concatenate([
                        entry["true_wld"][:, target_idx] for entry, _ in projected
                    ])
                    color_min = float(np.min(target_values))
                    color_max = float(np.max(target_values))
                    fig, axes = plt.subplots(
                        nrows,
                        ncols,
                        figsize=(8.5 * ncols, 6.5 * nrows),
                        dpi=300,
                        squeeze=False,
                        sharex=True,
                        sharey=True,
                    )
                    scatter = None
                    cmap_choice = reg_cmaps.get(target_name, "viridis")
                    for ax, (entry, projection) in zip(axes.flat, projected):
                        scatter = ax.scatter(
                            projection[:, 0],
                            projection[:, 1],
                            c=entry["true_wld"][:, target_idx],
                            cmap=cmap_choice,
                            s=55,
                            alpha=0.85,
                            edgecolors="black",
                            linewidth=0.3,
                            vmin=color_min,
                            vmax=color_max,
                        )
                        ax.set_title(entry["config_label"], fontsize=15, fontweight="bold", pad=8)
                        ax.set_xlabel(f"Joint {method_label} Dim 1", fontsize=14, fontweight="bold")
                        ax.set_ylabel(f"Joint {method_label} Dim 2", fontsize=14, fontweight="bold")
                        ax.tick_params(axis="both", which="major", labelsize=12)
                        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)
                    for ax in axes.flat[n_panels:]:
                        ax.axis("off")
                    if scatter is not None:
                        cbar = fig.colorbar(
                            scatter,
                            ax=axes.ravel().tolist(),
                            pad=0.03,
                            fraction=0.03,
                        )
                        cbar.set_label(f"True {target_name} ({unit_map.get(target_name, 'mm')})", fontsize=14, fontweight="bold")
                        cbar.ax.tick_params(labelsize=12)
                    fig.suptitle(
                        f"Regression Embedding — Joint {method_label} Colored by {target_name} ({unit_map.get(target_name, 'mm')}) (Train {percent}%)",
                        fontsize=18,
                        fontweight="bold",
                        y=0.98,
                    )
                    fig.tight_layout(rect=(0, 0, 0.95, 0.95))
                    for ext in (".png", ".pdf"):
                        fig.savefig(
                            os.path.join(
                                feature_dir,
                                f"regression_joint_{method_name}_colored_by_{target_name.lower()}{ext}",
                            ),
                            dpi=300,
                            bbox_inches="tight",
                        )
                    plt.close(fig)

    separation_df = pd.DataFrame(separation_rows)
    for feature_space in separation_df["feature_space"].unique():
        baseline = separation_df[
            (separation_df["feature_space"] == feature_space)
            & (separation_df["mode"] == "no_pinn")
        ]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        mask = separation_df["feature_space"] == feature_space
        for metric in [
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "mean_class_centroid_distance",
        ]:
            separation_df.loc[mask, f"{metric}_delta_vs_no_pinn"] = (
                separation_df.loc[mask, metric] - float(baseline_row[metric])
            )
        # Positive means PINN improved the metric relative to No-PINN.
        separation_df.loc[mask, "davies_bouldin_improvement_vs_no_pinn"] = (
            float(baseline_row["davies_bouldin"])
            - separation_df.loc[mask, "davies_bouldin"]
        )
    _atomic_to_csv(
        separation_df,
        os.path.join(feature_dir, "feature_separation_metrics.csv"),
        index=False,
    )
    for metric, label, higher_is_better in [
        ("silhouette", "Silhouette score", True),
        ("calinski_harabasz", "Calinski-Harabasz score", True),
        ("davies_bouldin", "Davies-Bouldin score", False),
        ("mean_class_centroid_distance", "Mean class centroid distance", True),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        configs = (
            separation_df[["config_label", "config_order"]]
            .drop_duplicates()
            .sort_values("config_order")
        )
        config_labels = configs["config_label"].tolist()
        x = np.arange(len(config_labels))
        spaces = [
            ("backbone", "Shared backbone"),
            ("regression", "Regression embedding"),
        ]
        width = 0.34
        for idx, (space_key, space_label) in enumerate(spaces):
            values = []
            for config_label in config_labels:
                match = separation_df[
                    (separation_df["config_label"] == config_label)
                    & (separation_df["feature_space"] == space_key)
                ]
                values.append(float(match.iloc[0][metric]) if not match.empty else np.nan)
            ax.bar(
                x + (idx - 0.5) * width,
                values,
                width=width,
                label=space_label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=15)
        ax.set_xlabel("Model configuration")
        direction = "higher" if higher_is_better else "lower"
        ax.set_ylabel(f"{label} - {direction} is better")
        ax.set_title(f"Feature separation: {label}", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(feature_dir, f"feature_{metric}_bar.png"),
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"[OK] Saved feature visualizations: {feature_dir}")


def save_warmup_metric_bar_plots(warmup_series, report_dir):
    """Compare fixed-alpha warmup runs with grouped metric bar charts."""
    warmup_series = sorted(
        [item for item in warmup_series if item.get("warmup") is not None],
        key=lambda item: int(item["warmup"]),
    )
    percent_values = sorted({
        pct for item in warmup_series for pct in item.get("results", {})
    })
    for pct in percent_values:
        completed = [
            item for item in warmup_series
            if item.get("results", {}).get(pct) is not None
        ]
        if not completed:
            continue

        out_dir = os.path.join(report_dir, "warmup_study", f"train_{pct:02d}pct")
        os.makedirs(out_dir, exist_ok=True)
        labels = [f"W={int(item['warmup'])}" for item in completed]
        records = [item["results"][pct] for item in completed]
        x = np.arange(len(completed))
        save_warmup_feature_visualizations(completed, records, out_dir, pct)

        rows = []
        for item, record in zip(completed, records):
            row = {
                "train_percent": pct,
                "alpha": item.get("alpha"),
                "warmup": item.get("warmup"),
                "seed": item.get("seed_list", [""])[0],
                "balanced_accuracy": record.get("clf_balanced_accuracy", np.nan),
                "macro_precision": record.get("clf_precision_macro", np.nan),
                "macro_recall": record.get("clf_recall_macro", np.nan),
                "macro_f1": record.get("clf_f1_macro", np.nan),
                "mcc": record.get("clf_mcc", np.nan),
                "mae_avg": record.get("mae_avg", np.nan),
                "rmse_avg": record.get("rmse_avg", np.nan),
                "r2_avg": record.get("r2_avg", np.nan),
                "nrmse_avg": record.get("nrmse_avg", np.nan),
                "bias_avg": record.get("mean_error_avg", np.nan),
                "mae_w": record.get("mae_w", np.nan),
                "mae_l": record.get("mae_l", np.nan),
                "mae_d": record.get("mae_d", np.nan),
                "rmse_w": record.get("rmse_w", np.nan),
                "rmse_l": record.get("rmse_l", np.nan),
                "rmse_d": record.get("rmse_d", np.nan),
                "r2_w": record.get("r2_w", np.nan),
                "r2_l": record.get("r2_l", np.nan),
                "r2_d": record.get("r2_d", np.nan),
                "nrmse_w": record.get("nrmse_w", np.nan),
                "nrmse_l": record.get("nrmse_l", np.nan),
                "nrmse_d": record.get("nrmse_d", np.nan),
                "bias_w": record.get("mean_error_w", np.nan),
                "bias_l": record.get("mean_error_l", np.nan),
                "bias_d": record.get("mean_error_d", np.nan),
                "run_dir": record.get("run_dir", ""),
            }
            rows.append(row)
        _atomic_to_csv(
            pd.DataFrame(rows),
            os.path.join(out_dir, "warmup_selected_metrics_summary.csv"),
            index=False,
        )

        per_class_frames = []
        for item, record in zip(completed, records):
            class_csv = os.path.join(record.get("run_dir", ""), "test_class_metrics.csv")
            if not os.path.isfile(class_csv):
                continue
            try:
                class_df = pd.read_csv(class_csv)
            except Exception as exc:
                print(f"[WARN] Failed reading per-class metrics {class_csv}: {exc}")
                continue
            required_counts = {"tp", "fp", "fn", "tn"}
            if required_counts.issubset(class_df.columns):
                tp = class_df["tp"].astype(float)
                fp = class_df["fp"].astype(float)
                fn = class_df["fn"].astype(float)
                tn = class_df["tn"].astype(float)
                sensitivity = tp / (tp + fn).replace(0, np.nan)
                specificity = tn / (tn + fp).replace(0, np.nan)
                class_df["balanced_accuracy_ovr"] = (
                    (sensitivity + specificity) / 2.0
                ).fillna(0.0)
                denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                numerator = tp * tn - fp * fn
                class_df["mcc_ovr"] = np.divide(
                    numerator,
                    denom,
                    out=np.zeros_like(numerator, dtype=float),
                    where=denom.to_numpy() != 0,
                )
            class_df["warmup"] = int(item["warmup"])
            class_df["alpha"] = item.get("alpha")
            class_df["seed"] = item.get("seed_list", [""])[0]
            class_df["source_csv"] = class_csv
            per_class_frames.append(class_df)

        if per_class_frames:
            per_class_df = pd.concat(per_class_frames, ignore_index=True)
            selected_class_cols = [
                "warmup", "alpha", "seed", "shape", "n_samples",
                "balanced_accuracy_ovr", "precision", "recall", "f1",
                "mcc_ovr", "tp", "fp", "fn", "tn", "source_csv",
            ]
            selected_class_cols = [
                col for col in selected_class_cols if col in per_class_df.columns
            ]
            _atomic_to_csv(
                per_class_df[selected_class_cols],
                os.path.join(out_dir, "warmup_per_class_metrics.csv"),
                index=False,
            )

            per_class_specs = [
                ("balanced_accuracy_ovr", "Balanced Accuracy", 100.0),
                ("precision", "Precision", 100.0),
                ("recall", "Recall", 100.0),
                ("f1", "F1", 100.0),
                ("mcc_ovr", "MCC", 1.0),
            ]
            class_names = sorted(per_class_df["shape"].dropna().unique())
            warmup_values = [int(item["warmup"]) for item in completed]
            class_width = 0.8 / max(len(class_names), 1)
            for metric_key, metric_label, scale in per_class_specs:
                if metric_key not in per_class_df.columns:
                    continue
                fig, ax = plt.subplots(figsize=(13, 7))
                for class_idx, class_name in enumerate(class_names):
                    values = []
                    for warmup in warmup_values:
                        match = per_class_df[
                            (per_class_df["warmup"] == warmup)
                            & (per_class_df["shape"] == class_name)
                        ]
                        value = float(match.iloc[0][metric_key]) if not match.empty else np.nan
                        values.append(value * scale)
                    offset = (
                        class_idx - (len(class_names) - 1) / 2.0
                    ) * class_width
                    ax.bar(
                        x + offset,
                        values,
                        width=class_width,
                        label=class_name,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_xlabel("PINN activation epoch (warmup)")
                unit = " (%)" if scale == 100.0 else ""
                ax.set_ylabel(f"{metric_label}{unit} - higher is better")
                ax.set_title(
                    f"Per-class {metric_label} by Warmup - train {pct}%, alpha=1, seed=42",
                    fontweight="bold",
                )
                ax.grid(axis="y", linestyle="--", alpha=0.45)
                ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(out_dir, f"classification_per_class_{metric_key}_bar.png"),
                    dpi=220,
                    bbox_inches="tight",
                )
                plt.close(fig)

        per_shape_frames = []
        for item, record in zip(completed, records):
            shape_csv = os.path.join(record.get("run_dir", ""), "test_metrics_per_shape.csv")
            if not os.path.isfile(shape_csv):
                continue
            try:
                shape_df = pd.read_csv(shape_csv)
            except Exception as exc:
                print(f"[WARN] Failed reading per-shape regression metrics {shape_csv}: {exc}")
                continue
            shape_df["warmup"] = int(item["warmup"])
            shape_df["alpha"] = item.get("alpha")
            shape_df["seed"] = item.get("seed_list", [""])[0]
            shape_df["source_csv"] = shape_csv
            per_shape_frames.append(shape_df)

        if per_shape_frames:
            per_shape_df = pd.concat(per_shape_frames, ignore_index=True)
            shape_metric_cols = [
                f"{metric}_{suffix}"
                for metric in ("mae", "rmse", "r2", "nrmse", "bias")
                for suffix in ("w", "l", "d")
            ]
            selected_shape_cols = [
                "warmup", "alpha", "seed", "shape", "n_samples",
                *shape_metric_cols, "source_csv",
            ]
            selected_shape_cols = [
                col for col in selected_shape_cols if col in per_shape_df.columns
            ]
            _atomic_to_csv(
                per_shape_df[selected_shape_cols],
                os.path.join(out_dir, "warmup_regression_by_class_and_wld.csv"),
                index=False,
            )

            regression_by_class_dir = os.path.join(out_dir, "regression_by_class")
            os.makedirs(regression_by_class_dir, exist_ok=True)
            warmup_values = [int(item["warmup"]) for item in completed]
            for class_name in sorted(per_shape_df["shape"].dropna().unique()):
                class_df = per_shape_df[per_shape_df["shape"] == class_name]
                safe_class = _safe_filename(class_name)
                for metric_key, metric_label, higher_is_better in [
                    ("mae", "MAE", False),
                    ("rmse", "RMSE", False),
                    ("r2", "R2", True),
                    ("nrmse", "NRMSE (%)", False),
                    ("bias", "Bias", False),
                ]:
                    target_keys = [
                        f"{metric_key}_{suffix}" for suffix in ("w", "l", "d")
                    ]
                    if not all(key in class_df.columns for key in target_keys):
                        continue
                    fig, ax = plt.subplots(figsize=(11, 6))
                    width = 0.24
                    for target_idx, (suffix, target_name) in enumerate(
                        zip(("w", "l", "d"), ("W", "L", "D"))
                    ):
                        values = []
                        for warmup in warmup_values:
                            match = class_df[class_df["warmup"] == warmup]
                            value = (
                                float(match.iloc[0][f"{metric_key}_{suffix}"])
                                if not match.empty else np.nan
                            )
                            values.append(value)
                        ax.bar(
                            x + (target_idx - 1) * width,
                            values,
                            width=width,
                            label=target_name,
                        )
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    ax.set_xlabel("PINN activation epoch (warmup)")
                    direction = "higher" if higher_is_better else "lower"
                    if metric_key == "bias":
                        direction = "closer to zero"
                        ax.axhline(0.0, color="black", linewidth=1.2)
                    ax.set_ylabel(f"{metric_label} - {direction} is better")
                    ax.set_title(
                        f"{class_name}: {metric_label} for W/L/D by Warmup",
                        fontweight="bold",
                    )
                    ax.grid(axis="y", linestyle="--", alpha=0.45)
                    ax.legend(title="Output")
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(
                            regression_by_class_dir,
                            f"{safe_class}_{metric_key}_wld_bar.png",
                        ),
                        dpi=220,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

        classification_specs = [
            ("clf_balanced_accuracy", "Balanced Accuracy", 100.0),
            ("clf_precision_macro", "Macro Precision", 100.0),
            ("clf_recall_macro", "Macro Recall", 100.0),
            ("clf_f1_macro", "Macro F1", 100.0),
            ("clf_mcc", "MCC", 1.0),
        ]
        for key, label, scale in classification_specs:
            values = np.asarray(
                [float(record.get(key, np.nan)) * scale for record in records],
                dtype=float,
            )
            if not np.isfinite(values).any():
                continue
            fig, ax = plt.subplots(figsize=(9, 6))
            bars = ax.bar(x, values, color=[item["color"] for item in completed])
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("PINN activation epoch (warmup)")
            unit = " (%)" if scale == 100.0 else ""
            ax.set_ylabel(f"{label}{unit} - higher is better")
            ax.set_title(f"{label} by Warmup - train {pct}%, alpha=1, seed=42", fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.45)
            ax.bar_label(bars, fmt="%.4g", padding=3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"classification_{key.replace('clf_', '')}_bar.png"),
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)

        for metric_key, metric_label, higher_is_better in [
            ("mae", "MAE (mm)", False),
            ("rmse", "RMSE (mm)", False),
            ("max_error", "Max Error (mm)", False),
            ("nrmse", "NRMSE (%)", False),
            ("r2", "R2 Score", True),
            ("mean_error", "Bias (mm)", False),
        ]:
            target_keys = [f"{metric_key}_{suffix}" for suffix in ("w", "l", "d")]
            if not any(
                any(np.isfinite(float(record.get(key, np.nan))) for record in records)
                for key in target_keys
            ):
                continue
            width = 0.24
            fig, ax = plt.subplots(figsize=(11, 6))
            for target_idx, (suffix, target_name) in enumerate(
                zip(("w", "l", "d"), ("W", "L", "D"))
            ):
                values = [
                    float(record.get(f"{metric_key}_{suffix}", np.nan))
                    for record in records
                ]
                ax.bar(x + (target_idx - 1) * width, values, width=width, label=target_name)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("PINN activation epoch (warmup)")
            direction = "higher" if higher_is_better else "lower"
            ax.set_ylabel(f"{metric_label} - {direction} is better")
            ax.set_title(
                f"{metric_label} for W/L/D - train {pct}%, alpha=1, seed=42",
                fontweight="bold",
            )
            if metric_key == "mean_error":
                ax.axhline(0.0, color="black", linewidth=1.2)
            ax.grid(axis="y", linestyle="--", alpha=0.45)
            ax.legend(title="Output")
            fig.tight_layout()
            file_metric = "bias" if metric_key == "mean_error" else metric_key
            fig.savefig(
                os.path.join(out_dir, f"regression_{file_metric}_wld_bar.png"),
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)

            avg_key = f"{metric_key}_avg"
            avg_values = np.asarray(
                [float(record.get(avg_key, np.nan)) for record in records],
                dtype=float,
            )
            if np.isfinite(avg_values).any():
                fig, ax = plt.subplots(figsize=(9, 6))
                bars = ax.bar(x, avg_values, color=[item["color"] for item in completed])
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_xlabel("PINN activation epoch (warmup)")
                direction = "higher" if higher_is_better else "lower"
                ax.set_ylabel(f"Average {metric_label} - {direction} is better")
                ax.set_title(
                    f"Average {metric_label} by Warmup - train {pct}%, alpha=1, seed=42",
                    fontweight="bold",
                )
                if metric_key == "mean_error":
                    ax.axhline(0.0, color="black", linewidth=1.2)
                ax.grid(axis="y", linestyle="--", alpha=0.45)
                ax.bar_label(bars, fmt="%.4g", padding=3)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(out_dir, f"regression_{file_metric}_average_bar.png"),
                    dpi=220,
                    bbox_inches="tight",
                )
                plt.close(fig)

        print(f"[OK] Saved warmup-study bar charts: {out_dir}")


def save_pinn_effect_feature_visualizations(series_items, report_dir):
    """Compare No-PINN and PINN feature spaces on the same projection basis."""
    comparison_series = [
        item for item in series_items
        if item.get("mode") == "no_pinn" or item.get("group") == "warmup"
    ]
    percent_values = sorted({
        pct for item in comparison_series for pct in item.get("results", {})
    })
    for pct in percent_values:
        completed = [
            item for item in comparison_series
            if item.get("results", {}).get(pct) is not None
        ]
        if not any(item.get("mode") == "no_pinn" for item in completed):
            continue
        if not any(item.get("mode") == "pinn" for item in completed):
            continue
        completed.sort(
            key=lambda item: (
                0 if item.get("mode") == "no_pinn" else 1,
                -1 if item.get("warmup") is None else int(item["warmup"]),
            )
        )
        records = [item["results"][pct] for item in completed]
        out_dir = os.path.join(
            report_dir,
            "pinn_effect_feature_comparison",
            f"train_{pct:02d}pct",
        )
        os.makedirs(out_dir, exist_ok=True)
        save_warmup_feature_visualizations(completed, records, out_dir, pct)
        print(f"[OK] Saved No-PINN vs PINN feature comparison: {out_dir}")


def save_seed_grouped_bar_plots(tgt_pcts, seed_series, report_dir, *, live=False):
    """Save paired No-PINN/PINN bars for each seed across all selected percents in a combined plot."""
    os.makedirs(report_dir, exist_ok=True)
    mode_specs = [
        ("no_pinn", "No PINN", "#2774AE"),
        ("pinn", "PINN", "#D1495B"),
    ]

    by_pct_seed = defaultdict(lambda: defaultdict(dict))
    all_seeds = set()
    rows = []
    
    for item in seed_series:
        mode = item.get("mode")
        label = item.get("label", "")
        per_seed = item.get("per_seed_results", {})
        for seed, res_dict in per_seed.items():
            seed_str = str(seed)
            for pct in tgt_pcts:
                rec = res_dict.get(pct)
                if rec is None:
                    continue
                if mode in by_pct_seed[pct][seed_str] and item.get("group") == "baseline":
                    continue
                by_pct_seed[pct][seed_str][mode] = rec
                all_seeds.add(seed_str)
                rows.append({
                    "train_percent": pct,
                    "seed": seed_str,
                    "mode": mode,
                    "series": label,
                    "mae_avg": rec.get("mae", rec.get("mae_avg", np.nan)),
                    "rmse_avg": rec.get("rmse", rec.get("rmse_avg", np.nan)),
                    "r2_avg": rec.get("r2", rec.get("r2_avg", np.nan)),
                    "nrmse_avg": rec.get("nrmse", rec.get("nrmse_avg", np.nan)),
                    "clf_acc_percent": rec.get("acc", rec.get("clf_acc_percent", np.nan)),
                    "clf_f1_macro": rec.get("f1_macro", rec.get("clf_f1_macro", np.nan)),
                    "clf_mcc": rec.get("mcc", rec.get("clf_mcc", np.nan)),
                    "run_dir": rec.get("run_dir", ""),
                    "csv_file": rec.get("csv_file", ""),
                })

    if not by_pct_seed:
        return

    def _seed_sort_key(seed_value):
        try:
            return (0, int(seed_value))
        except Exception:
            return (1, str(seed_value))

    sorted_seeds = sorted(all_seeds, key=_seed_sort_key)
    
    for pct in tgt_pcts:
        pct_rows = [r for r in rows if r["train_percent"] == pct]
        if pct_rows:
            table_path = os.path.join(report_dir, f"seed_metrics_train_{pct:02d}pct.csv")
            _atomic_to_csv(pd.DataFrame(pct_rows), table_path, index=False)

    plot_specs = [
        (
            "rmse",
            "Seed Test RMSE",
            "Average RMSE (mm) - Lower is better",
            "seed_rmse_grouped_bar_combined.png",
        ),
        (
            "mae",
            "Seed Test MAE",
            "Average MAE (mm) - Lower is better",
            "seed_mae_grouped_bar_combined.png",
        ),
        (
            "nrmse",
            "Seed Test NRMSE",
            "Average NRMSE (%) - Lower is better",
            "seed_nrmse_grouped_bar_combined.png",
        ),
        (
            "r2",
            "Seed Test R² Score",
            "Average R² - Higher is better",
            "seed_r2_grouped_bar_combined.png",
        ),
        (
            "acc",
            "Seed Test Classification Accuracy",
            "Accuracy (%) - Higher is better",
            "seed_acc_grouped_bar_combined.png",
        ),
        (
            "f1_macro",
            "Seed Test Macro F1",
            "Macro F1 (%) - Higher is better",
            "seed_f1_macro_grouped_bar_combined.png",
        ),
        (
            "mcc",
            "Seed Test MCC",
            "Matthews Correlation Coefficient - Higher is better",
            "seed_mcc_grouped_bar_combined.png",
        ),
    ]

    x = np.arange(len(sorted_seeds), dtype=float)
    width = 0.36
    n_pcts = len(tgt_pcts)

    for metric_key, title, y_label, filename in plot_specs:
        fig, axes = plt.subplots(1, n_pcts, figsize=(4.5 * n_pcts + 2, 6), sharey=False, squeeze=False)
        axes = axes[0]
        global_plotted = False
        
        for ax_idx, pct in enumerate(tgt_pcts):
            ax = axes[ax_idx]
            pct_data = by_pct_seed[pct]
            
            for mode_idx, (mode, mode_label, color) in enumerate(mode_specs):
                offset = (mode_idx - 0.5) * width
                values = [
                    float(pct_data[seed].get(mode, {}).get(metric_key, np.nan))
                    for seed in sorted_seeds
                ]
                bars = ax.bar(
                    x + offset,
                    values,
                    width,
                    label=mode_label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
                for bar, value in zip(bars, values):
                    if not np.isfinite(value):
                        continue
                    global_plotted = True
                    ax.annotate(
                        f"{value:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            
            ax.set_title(f"Train {pct}%", fontsize=12, fontweight="bold")
            ax.set_xlabel("Random Seed", fontsize=10, fontweight="bold")
            if ax_idx == 0:
                ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_seeds)
            ax.grid(axis="y", linestyle="--", alpha=0.45)
            
        prefix = "LIVE " if live else ""
        fig.suptitle(f"{prefix}{title} Across All Seeds", fontsize=14, fontweight="bold", y=0.98)
        
        if global_plotted:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.95))
            
        fig.tight_layout(rect=[0, 0, 0.88, 0.93])
        fig.savefig(os.path.join(report_dir, filename), dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Saved combined seed grouped bar charts to: {report_dir}")


def refresh_live_comparison_plots(series_dict, report_dir):
    """Refresh comparison plots from current completed outputs.

    Used for real-time monitoring while training is still running.
    """
    all_percents = set()

    for item in series_dict:
        per_seed_results = {}
        seed_list = item.get("seed_list", SEEDS)
        allowed_percents = {
            int(p) for p in item.get("percent_list", PERCENT_LIST)
        }
        for seed in seed_list:
            seed_suffix = f"{item['tag']}_seed_{seed}"
            res = {
                pct: rec
                for pct, rec in extract_metrics(item["mode"], seed_suffix).items()
                if int(pct) in allowed_percents
            }
            per_seed_results[seed] = res
            all_percents.update(res.keys())
        item["per_seed_results"] = per_seed_results
        item["results"] = aggregate_seed_results(per_seed_results)

    target_percents = sorted(all_percents)
    if not target_percents:
        print("[LIVE] No completed runs yet, skipping live plot refresh.")
        return False

    live_dir = os.path.join(report_dir, "live_progress")
    os.makedirs(live_dir, exist_ok=True)
    main_series = [s for s in series_dict if s.get("group") not in ("seed", "warmup")]
    seed_series = [s for s in series_dict if s.get("group") == "seed"]
    if not seed_series:
        alphas = [s for s in series_dict if s.get("group") == "alpha"]
        if alphas:
            a1 = [a for a in alphas if a.get("alpha") == 1.0]
            seed_series.append(a1[0] if a1 else alphas[-1])
    seed_series += [s for s in series_dict if s.get("group") == "baseline"]

    if main_series:
        main_percents = sorted({
            pct for s in main_series for pct in s.get("results", {}).keys()
        })
        if main_percents:
            main_dir = os.path.join(live_dir, "main")
            os.makedirs(main_dir, exist_ok=True)
            live_line_plots = [
                ("mae", "LIVE Average Test MAE", "Average Test MAE (mm) $\\downarrow$", "comparison_mae_live.png", True),
                ("rmse", "LIVE Average Test RMSE", "Average Test RMSE (mm) $\\downarrow$", "comparison_rmse_live.png", True),
                ("r2", "LIVE Test R² Score", "Test $R^2$ Score $\\uparrow$", "comparison_r2_live.png", False),
                ("max_error", "LIVE Maximum Error", "Maximum Error (mm) $\\downarrow$", "comparison_max_error_live.png", True),
                ("nrmse", "LIVE Normalized RMSE", "Normalized RMSE (%) $\\downarrow$", "comparison_nrmse_live.png", True),
                ("acc", "LIVE Classification Accuracy", "Classification Accuracy (%) $\\uparrow$", "comparison_acc_live.png", False),
                ("balanced_accuracy", "LIVE Balanced Accuracy", "Balanced Accuracy (%) $\\uparrow$", "comparison_balanced_acc_live.png", False),
                ("f1_macro", "LIVE Macro F1-Score", "Macro F1-Score (%) $\\uparrow$", "comparison_f1_macro_live.png", False),
                ("mcc", "LIVE Matthews Correlation Coefficient", "Matthews Correlation Coefficient (MCC) $\\uparrow$", "comparison_mcc_live.png", False),
            ]
            for m_key, m_title, m_ylabel, m_fname, m_lower in live_line_plots:
                save_line_plot(
                    main_percents,
                    main_series,
                    metric_key=m_key,
                    title=m_title,
                    y_label=m_ylabel,
                    out_file=os.path.join(main_dir, m_fname),
                    lower_is_better=m_lower,
                )
            save_percent_bar_plots(main_percents, main_series, main_dir)

    if seed_series:
        seed_percents = sorted({
            pct for s in seed_series for pct in s.get("results", {}).keys()
        })
        if seed_percents:
            seed_dir = os.path.join(live_dir, "seed_test")
            os.makedirs(seed_dir, exist_ok=True)
            save_seed_grouped_bar_plots(seed_percents, seed_series, seed_dir, live=True)

    print(f"[LIVE] Refreshed live comparison plots in: {live_dir}")
    return True


def _extract_metric_csv_meta(csv_path):
    path_parts = csv_path.replace("\\", "/").split("/")
    mode = "PINN" if "Outputs_cnn_pinn" in csv_path else "NO_PINN"
    train_pct = None
    seed = None
    for part in path_parts:
        if part.startswith("train_") and part.endswith("pct"):
            train_pct = int(part.split("_")[1].replace("pct", ""))
        if "seed_" in part:
            try:
                seed = int(part.split("seed_")[1].split("_")[0])
            except Exception:
                seed = None
    return mode, train_pct, seed


def consolidate_class_and_target_metrics(report_dir):
    specs = [
        {
            "filename": "test_classification_metrics_overall.csv",
            "combined": "all_classification_metrics_overall.csv",
            "summary": "classification_metrics_overall_summary_stats.csv",
            "group_cols": ["mode", "train_percent"],
            "agg_cols": [
                "clf_accuracy", "clf_balanced_accuracy",
                "clf_precision_macro", "clf_recall_macro", "clf_f1_macro",
                "clf_precision_micro", "clf_recall_micro", "clf_f1_micro",
                "clf_precision_weighted", "clf_recall_weighted", "clf_f1_weighted",
                "clf_mcc", "clf_cohen_kappa",
            ],
        },
        {
            "filename": "test_class_metrics.csv",
            "combined": "all_class_metrics.csv",
            "summary": "class_metrics_summary_stats.csv",
            "group_cols": ["mode", "train_percent", "shape"],
            "agg_cols": [
                "class_accuracy", "class_accuracy_percent", "precision",
                "recall", "f1", "specificity", "balanced_accuracy_ovr",
                "mcc_ovr", "one_vs_rest_accuracy",
                "n_samples",
            ],
        },
        {
            "filename": "test_regression_metrics_by_target.csv",
            "combined": "all_regression_metrics_by_target.csv",
            "summary": "regression_metrics_by_target_summary_stats.csv",
            "group_cols": ["mode", "train_percent", "target"],
            "agg_cols": [
                "mae", "mse", "rmse", "r2", "max_error", "nrmse", "mean_error",
                "median_ae", "explained_variance",
                "n_samples",
            ],
        },
    ]

    for spec in specs:
        files = []
        for pattern in [
            os.path.join(OUTPUT_NO_PINN_ROOT, "**", spec["filename"]),
            os.path.join(OUTPUT_PINN_ROOT, "**", spec["filename"]),
        ]:
            files.extend(glob.glob(pattern, recursive=True))
        files = filter_preferred_metric_csvs(files)

        frames = []
        for csv_path in files:
            try:
                df = pd.read_csv(csv_path)
                mode, train_pct, seed = _extract_metric_csv_meta(csv_path)
                if train_pct not in PERCENT_LIST:
                    continue
                df["mode"] = mode
                df["train_percent"] = train_pct
                df["seed"] = seed
                df["source_csv"] = csv_path
                frames.append(df)
            except Exception as exc:
                print(f"[WARN] Failed reading {csv_path}: {exc}")

        if not frames:
            print(f"[INFO] No {spec['filename']} files found to consolidate.")
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined_csv = os.path.join(report_dir, spec["combined"])
        _atomic_to_csv(combined, combined_csv, index=False)
        print(f"[OK] Saved combined metrics: {combined_csv}")

        available_agg_cols = [c for c in spec["agg_cols"] if c in combined.columns]
        if available_agg_cols:
            summary = combined.groupby(spec["group_cols"])[available_agg_cols].agg(["mean", "std"]).round(6)
            summary_csv = os.path.join(report_dir, spec["summary"])
            _atomic_to_csv(summary, summary_csv)
            print(f"[OK] Saved metrics summary: {summary_csv}")


def cohen_d(x, y):
    """Compute Cohen's d effect size."""
    nx = len(x); ny = len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    pooled = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    if pooled <= 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / math.sqrt(pooled)


def summary_stats(arr):
    """Compute mean, std, 95% CI."""
    a = np.array(arr, dtype=float)
    n = len(a)
    mean = np.mean(a)
    sd = np.std(a, ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * se
    return mean, sd, ci95


def extract_meta_from_path(path):
    """Extract config, percent, seed from file path."""
    percent = None
    seed = None
    parts = path.replace('\\', '/').split('/')
    for part in parts:
        if part.startswith('train_') and part.endswith('pct'):
            try:
                percent = int(part.split('_')[-1].replace('pct',''))
            except Exception:
                pass
        if 'seed_' in part:
            try:
                seed = int(part.split('seed_')[-1].split('_')[0])
            except Exception:
                pass
    
    if os.path.abspath(path).startswith(os.path.abspath(OUTPUT_PINN_ROOT) + os.sep):
        config = 'PINN'
    elif 'Outputs_cnn_baseline' in path or 'Outputs_cnn_no_pinn' in path:
        config = 'NO_PINN'
    else:
        config = 'UNKNOWN'
    return config, percent, seed


def aggregate_detailed_metrics(report_dir):
    """Search for detailed_metrics.csv in output folders and compute per-percent statistics."""
    patterns = [
        os.path.join(OUTPUT_NO_PINN_ROOT, '**', 'detailed_metrics.csv'),
        os.path.join(OUTPUT_PINN_ROOT, '**', 'detailed_metrics.csv'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    
    if not files:
        print(f"[INFO] No detailed_metrics.csv files found. Skipping aggregation.")
        return
    
    print(f"[INFO] Found {len(files)} detailed_metrics.csv files")
    
    records = defaultdict(lambda: {})  # (config, percent, seed) -> row
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.shape[0] == 0:
                continue
            row = df.iloc[-1].to_dict()
            cfg, pct, seed = extract_meta_from_path(f)
            key = (cfg, pct, seed)
            records[key] = row
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")
            continue
    
    per_percent = defaultdict(lambda: defaultdict(dict))
    for (cfg, pct, seed), row in records.items():
        per_percent[pct][cfg][seed] = row
    
    rows_out = []
    for pct in sorted(per_percent.keys()):
        pinn_seeds = sorted(per_percent[pct].get('PINN', {}).keys())
        nopinn_seeds = sorted(per_percent[pct].get('NO_PINN', {}).keys())
        common_seeds = sorted(set(pinn_seeds) & set(nopinn_seeds))
        if not common_seeds:
            print(f"[WARN] No paired runs for percent {pct}")
            continue
        
        sample_row = per_percent[pct]['PINN'][common_seeds[0]]
        chosen_metric = None
        for cand in ['MAE_W', 'RMSE_W', 'MAE', 'RMSE', 'MAE_total', 'mae_avg', 'rmse_avg']:
            if cand in sample_row:
                chosen_metric = cand
                break
        if chosen_metric is None:
            for k, v in sample_row.items():
                try:
                    float(v)
                    chosen_metric = k
                    break
                except Exception:
                    continue
        if chosen_metric is None:
            print(f"[WARN] Could not find numeric metric for percent {pct}")
            continue
        
        pinn_vals = np.array([float(per_percent[pct]['PINN'][s].get(chosen_metric, float('nan'))) for s in common_seeds])
        nopinn_vals = np.array([float(per_percent[pct]['NO_PINN'][s].get(chosen_metric, float('nan'))) for s in common_seeds])
        
        mean_p, sd_p, ci_p = summary_stats(pinn_vals)
        mean_n, sd_n, ci_n = summary_stats(nopinn_vals)
        
        t_stat, t_p = (None, None)
        w_stat, w_p = (None, None)
        if len(common_seeds) > 1:
            try:
                t_res = stats.ttest_rel(pinn_vals, nopinn_vals, nan_policy='omit')
                t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
            except Exception:
                pass
            try:
                w_res = stats.wilcoxon(pinn_vals, nopinn_vals)
                w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                pass
        
        d = cohen_d(pinn_vals, nopinn_vals)
        
        rows_out.append({
            'percent': pct,
            'metric': chosen_metric,
            'n_pairs': len(common_seeds),
            'pinn_mean': mean_p,
            'pinn_sd': sd_p,
            'pinn_ci95': ci_p,
            'nopinn_mean': mean_n,
            'nopinn_sd': sd_n,
            'nopinn_ci95': ci_n,
            'ttest_stat': t_stat,
            'ttest_pvalue': t_p,
            'wilcoxon_stat': w_stat,
            'wilcoxon_pvalue': w_p,
            'cohens_d': d,
        })
    
    if rows_out:
        df_out = pd.DataFrame(rows_out)
        agg_csv = os.path.join(report_dir, 'aggregated_detailed_metrics_stats.csv')
        _atomic_to_csv(df_out, agg_csv, index=False)
        print(f"[OK] Saved aggregated stats: {agg_csv}")
    else:
        print("[INFO] No aggregated statistics computed.")


def organize_paper_plots(report_dir):
    """Organize all generated PNG images into dedicated, easy-to-find subfolders inside report_dir."""
    import shutil
    import glob
    paper_dir = os.path.join(report_dir, "paper_plots")
    v_dir = os.path.join(paper_dir, "violin_plots")
    r_dir = os.path.join(paper_dir, "regression_plots")
    c_dir = os.path.join(paper_dir, "classification_plots")
    
    os.makedirs(v_dir, exist_ok=True)
    os.makedirs(r_dir, exist_ok=True)
    os.makedirs(c_dir, exist_ok=True)
    
    src_violin_dir = os.path.join(report_dir, "violin_plots")
    if os.path.isdir(src_violin_dir):
        for f in os.listdir(src_violin_dir):
            if f.endswith(".png"):
                shutil.copy2(os.path.join(src_violin_dir, f), os.path.join(v_dir, f))
                
    for fname in os.listdir(report_dir):
        if not fname.endswith(".png"):
            continue
        src = os.path.join(report_dir, fname)
        if not os.path.isfile(src):
            continue
            
        lower_name = fname.lower()
        if "violin" in lower_name:
            shutil.copy2(src, os.path.join(v_dir, fname))
        elif any(k in lower_name for k in ["mae", "rmse", "r2", "error", "nrmse", "max_error", "regression", "residual", "test_curves"]):
            shutil.copy2(src, os.path.join(r_dir, fname))
        elif any(k in lower_name for k in ["clf", "class", "acc", "precision", "recall", "kappa"]):
            shutil.copy2(src, os.path.join(c_dir, fname))

    for mode_root, mode_name in [(OUTPUT_PINN_ROOT, "PINN"), (OUTPUT_NO_PINN_ROOT, "NO_PINN")]:
        if not os.path.isdir(mode_root):
            continue
        for fig3_path in glob.glob(os.path.join(mode_root, "**", "fig3_regression_scatter.png"), recursive=True):
            mode, train_pct, seed = _extract_metric_csv_meta(fig3_path)
            if train_pct is not None:
                seed_str = f"_seed_{seed}" if seed is not None else ""
                dest_name = f"fig3_regression_scatter_{mode_name}_train_{train_pct:02d}pct{seed_str}.png"
                shutil.copy2(fig3_path, os.path.join(r_dir, dest_name))
        for fig2_path in glob.glob(os.path.join(mode_root, "**", "fig2_confusion_matrix.png"), recursive=True):
            mode, train_pct, seed = _extract_metric_csv_meta(fig2_path)
            if train_pct is not None:
                seed_str = f"_seed_{seed}" if seed is not None else ""
                dest_name = f"fig2_confusion_matrix_{mode_name}_train_{train_pct:02d}pct{seed_str}.png"
                shutil.copy2(fig2_path, os.path.join(c_dir, dest_name))
            
    print(f"\n[OK] Successfully organized paper plots into dedicated subfolders:\n  - Violin Plots: {v_dir}\n  - Regression Plots: {r_dir}\n  - Classification Plots: {c_dir}\n")


def generate_manuscript_reports():
    """Generate publication-ready tables and figures for the paper."""
    print("\n" + "=" * 80)
    print("GENERATING MANUSCRIPT REPORTS")
    print("=" * 80)
    
    try:
        print("[INFO] Generating statistical tests table...")
        agg_stats_file = os.path.join(REPORT_DIR, 'aggregated_detailed_metrics_stats.csv')
        if os.path.exists(agg_stats_file):
            df_stats = pd.read_csv(agg_stats_file)
            print(f"[OK] Stats table already saved: {agg_stats_file}")
    except Exception as e:
        print(f"[WARN] Failed to generate stats table: {e}")
    
    try:
        organize_paper_plots(REPORT_DIR)
    except Exception as e:
        print(f"[WARN] Failed to organize paper plots: {e}")
        
    print("[OK] Manuscript report generation complete")


if __name__ == "__main__":
    if PLOT_METRICS_ONLY:
        metrics_path = os.path.join(REPORT_DIR, "all_metrics_long.csv")
        if not os.path.isfile(metrics_path):
            latest_metrics_path = os.path.join(
                REPORT_ROOT,
                "latest_all_metrics_long.csv",
            )
            if os.path.isfile(latest_metrics_path):
                metrics_path = latest_metrics_path
                print(
                    "[PLOT] Run-local metrics CSV is unavailable; "
                    f"using latest snapshot: {metrics_path}"
                )
            else:
                raise FileNotFoundError(
                    "Metrics CSV not found in run directory or latest snapshot"
                )
        metrics_df = pd.read_csv(metrics_path)
        _atomic_to_csv(
            metrics_df,
            os.path.join(REPORT_DIR, "all_metrics_long.csv"),
            index=False,
        )
        save_full_metric_tables(metrics_df, REPORT_DIR)
        print(f"[DONE] Metric tables refreshed from: {metrics_path}")
        sys.exit(0)

    print("=" * 80)
    print("RUN EVAL ALL: SPLIT BASELINE + ALPHA/WARMUP/SEED TESTS")
    print("=" * 80)
    print(f"[CONFIG] Evaluation preset: {EVAL_PRESET or 'default'}")
    print(f"[CONFIG] Percents: {PERCENT_LIST}")
    print(f"[CONFIG] Epochs: {EPOCHS}")
    print(f"[CONFIG] Parallel runs: {PARALLEL_RUNS}")
    print(f"[CONFIG] GPU ids: {', '.join(GPU_IDS)}")
    if PARALLEL_RUNS > len(GPU_IDS):
        print("[CONFIG][WARN] PARALLEL_RUNS is greater than GPU_IDS count; multiple jobs will share a GPU.")
    print(f"[CONFIG] PINN activation epoch: {PINN_ACTIVATION_EPOCH}")
    print(f"[CONFIG] PINN loss type folder: loss_{PINN_LOSS_TYPE}")
    print(f"[CONFIG] Alpha values: {', '.join(alpha_to_label(a) for a in ALPHA_VALUES)}")
    # Recalibrate alpha first, then run the seed study at one representative
    # train percentage using the selected alpha.
    test_groups = _parse_group_values(os.environ.get("TEST_GROUPS", "no_pinn,alpha,seed"))
    print(f"[CONFIG] Enabled test groups: {', '.join(test_groups)}")
    print(f"[CONFIG] Run training: {RUN_TRAIN}")
    print(f"[CONFIG] Skip existing results: {SKIP_IF_EXISTS}")
    print(f"[CONFIG] Auto-resume incomplete runs: {AUTO_RESUME}")
    print(f"[CONFIG] Reuse completed No-PINN results: {REUSE_COMPLETED_NO_PINN}")
    print(f"[CONFIG] Re-eval existing models: {REEVAL_IF_MODEL_EXISTS}")
    print(f"[CONFIG] Report re-eval only: {REPORT_REEVAL_ONLY}")
    if REPORT_REEVAL_ONLY and not REEVAL_IF_MODEL_EXISTS:
        print("[CONFIG][WARN] REPORT_REEVAL_ONLY=1 but REEVAL_IF_MODEL_EXISTS=0; report will not use stale CSV results.")
    print(f"[CONFIG] Live progress plots: {LIVE_PROGRESS_PLOTS}")
    print(f"[CONFIG] Infer before train: {INFER_BEFORE_TRAIN}")
    print(f"[CONFIG] Output root: {OUTPUT_ROOT}")
    print(f"[CONFIG] Study root: {STUDY_ROOT}")
    print(f"[CONFIG] PINN model root: {PINN_MODEL_ROOT}")
    print(f"[CONFIG] No-PINN model root: {NO_PINN_MODEL_ROOT}")
    print(f"[CONFIG] Report dir: {REPORT_DIR}")

    # ---------------------------------------------------------------------
    # GROUP-WISE TRAIN CONFIG (edit each block independently)
    # ---------------------------------------------------------------------
    # 1) BASELINE: No-PINN + 1 PINN baseline config
    baseline_alpha = float(os.environ.get("BASELINE_ALPHA", "1.0"))
    baseline_warmup = int(os.environ.get("BASELINE_WARMUP", "100"))
    baseline_percents = _parse_int_list(
        os.environ.get("BASELINE_PERCENTS", DEFAULT_PERCENT_LIST),
        "BASELINE_PERCENTS",
        min_value=1,
        max_value=100,
    )
    baseline_seeds = _parse_seed_list(os.environ.get("BASELINE_SEEDS", "42"))

    # Best PINN configuration used for warmup/seed studies.
    # By default it follows the baseline config but can be overridden independently.
    best_alpha_env_set = "BEST_ALPHA" in os.environ
    warmup_alpha_env_set = "WARMUP_TEST_ALPHA" in os.environ
    seed_alpha_env_set = "SEED_TEST_ALPHA" in os.environ
    auto_select_best_alpha = _get_bool_env("AUTO_SELECT_BEST_ALPHA", True)

    best_alpha = float(os.environ.get("BEST_ALPHA", f"{baseline_alpha:.12g}"))
    best_warmup = int(os.environ.get("BEST_WARMUP", str(baseline_warmup)))

    # 2) ALPHA TEST: PINN only, fixed warmup
    alpha_test_values = _parse_alpha_values(
        os.environ.get("ALPHA_TEST_VALUES", ",".join(alpha_to_label(a) for a in ALPHA_VALUES))
    )
    alpha_test_warmup = int(os.environ.get("ALPHA_TEST_WARMUP", str(best_warmup)))
    alpha_test_percents = _parse_int_list(
        os.environ.get("ALPHA_TEST_PERCENTS", DEFAULT_PERCENT_LIST),
        "ALPHA_TEST_PERCENTS",
        min_value=1,
        max_value=100,
    )
    alpha_test_seeds = _parse_seed_list(os.environ.get("ALPHA_TEST_SEEDS", "42"))



    # 4) SEED TEST: No-PINN + PINN at fixed alpha/warmup
    seed_test_alpha = float(os.environ.get("SEED_TEST_ALPHA", f"{best_alpha:.12g}"))
    seed_test_warmup = int(os.environ.get("SEED_TEST_WARMUP", str(best_warmup)))
    seed_test_percents = _parse_int_list(
        os.environ.get("SEED_TEST_PERCENTS", DEFAULT_SEED_TEST_PERCENT),
        "SEED_TEST_PERCENTS",
        min_value=1,
        max_value=100,
    )
    seed_test_seeds = _parse_seed_list(os.environ.get("SEED_TEST_SEEDS", "42,123,456,789"))

    print(f"[CONFIG][baseline] alpha={alpha_to_label(baseline_alpha)} warmup={baseline_warmup} percents={baseline_percents} seeds={baseline_seeds}")
    print(f"[CONFIG][best] alpha={alpha_to_label(best_alpha)} warmup={best_warmup} auto_select_alpha={auto_select_best_alpha and not best_alpha_env_set}")
    print(f"[CONFIG][alpha] values={[alpha_to_label(a) for a in alpha_test_values]} warmup={alpha_test_warmup} percents={alpha_test_percents} seeds={alpha_test_seeds}")

    print(f"[CONFIG][seed] alpha={alpha_to_label(seed_test_alpha)} warmup={seed_test_warmup} percents={seed_test_percents} seeds={seed_test_seeds}")

    series = []
    if "baseline" in test_groups or "no_pinn" in test_groups:
        series.append(
            make_series_item(
                key="no_pinn_baseline",
                label=f"No PINN baseline (E={EPOCHS})",
                short_label="No PINN",
                mode="no_pinn",
                script=SCRIPT_NO_PINN,
                tag=f"NoPINN_E{EPOCHS}",
                env={
                    "_MODE": "no_pinn",
                    "EPOCHS": str(EPOCHS),
                    "SERIES_LABEL": f"No PINN baseline (E={EPOCHS})",
                },
                color="#1f77b4",
                linestyle="--",
                marker="o",
                percent_list=baseline_percents,
                seed_list=baseline_seeds,
                group="baseline",
            )
        )

    if "baseline" in test_groups:
        series.append(
            make_series_item(
                key="pinn_baseline",
                label=f"PINN baseline a={alpha_to_label(baseline_alpha)} W={baseline_warmup}",
                short_label=f"PINN\na={alpha_to_label(baseline_alpha)} W={baseline_warmup}",
                mode="pinn",
                script=SCRIPT_PINN,
                tag=f"PINN_base_a{alpha_to_tag(baseline_alpha)}_W{baseline_warmup}_E{EPOCHS}",
                env={
                    "_MODE": "pinn",
                    "EPOCHS": str(EPOCHS),
                    "PINN_ACTIVATION_EPOCH": str(baseline_warmup),
                    "ALPHA_INIT": f"{baseline_alpha:.12g}",
                    "SERIES_LABEL": f"PINN baseline a={alpha_to_label(baseline_alpha)} W={baseline_warmup}",
                },
                color="#d62728",
                linestyle="-",
                marker="s",
                percent_list=baseline_percents,
                seed_list=baseline_seeds,
                alpha=baseline_alpha,
                warmup=baseline_warmup,
                group="baseline",
            )
        )

    palette = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]

    # Alpha test: PINN only, one percent and one seed.
    if "alpha" in test_groups:
        for idx, alpha in enumerate(alpha_test_values, start=1):
            a_label = alpha_to_label(alpha)
            series.append(
                make_series_item(
                    key=f"alpha_test_a{alpha_to_tag(alpha)}",
                    label=f"PINN alpha={a_label} W={alpha_test_warmup}",
                    short_label=f"a={a_label}",
                    mode="pinn",
                    script=SCRIPT_PINN,
                    tag=f"PINN_alpha_a{alpha_to_tag(alpha)}_W{alpha_test_warmup}_E{EPOCHS}",
                    env={
                        "_MODE": "pinn",
                        "EPOCHS": str(EPOCHS),
                        "PINN_ACTIVATION_EPOCH": str(alpha_test_warmup),
                        "ALPHA_INIT": f"{alpha:.12g}",
                        "SERIES_LABEL": f"PINN alpha={a_label} W={alpha_test_warmup}",
                    },
                    color=palette[(idx - 1) % len(palette)],
                    linestyle="-",
                    marker="s",
                    percent_list=alpha_test_percents,
                    seed_list=alpha_test_seeds,
                    alpha=alpha,
                    warmup=alpha_test_warmup,
                    group="alpha",
                )
            )


    # Seed test: keep each seed as its own series so plots/tables show seed id.
    if "seed" in test_groups:
        seed_colors = ["#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f", "#1f77b4"]
        seed_markers = ["o", "s", "^", "D", "v", "P"]
        for idx, seed in enumerate(seed_test_seeds):
            seed_str = str(seed)
            no_pinn_color = seed_colors[(2 * idx) % len(seed_colors)]
            pinn_color = seed_colors[(2 * idx + 1) % len(seed_colors)]
            marker = seed_markers[idx % len(seed_markers)]
            series.append(
                make_series_item(
                    key=f"no_pinn_seed_{seed_str}",
                    label=f"No PINN seed={seed_str}",
                    short_label=f"No PINN\nseed={seed_str}",
                    mode="no_pinn",
                    script=SCRIPT_NO_PINN,
                    tag=f"NoPINN_seed_E{EPOCHS}",
                    env={
                        "_MODE": "no_pinn",
                        "EPOCHS": str(EPOCHS),
                        "SERIES_LABEL": f"No PINN seed={seed_str} (E={EPOCHS})",
                    },
                    color=no_pinn_color,
                    linestyle="--",
                    marker=marker,
                    percent_list=seed_test_percents,
                    seed_list=[seed_str],
                    group="seed",
                )
            )
            series.append(
                make_series_item(
                    key=f"pinn_seed_{seed_str}",
                    label=f"PINN seed={seed_str} a={alpha_to_label(seed_test_alpha)} W={seed_test_warmup}",
                    short_label=f"PINN\nseed={seed_str}",
                    mode="pinn",
                    script=SCRIPT_PINN,
                    tag=f"PINN_seed_a{alpha_to_tag(seed_test_alpha)}_W{seed_test_warmup}_E{EPOCHS}",
                    env={
                        "_MODE": "pinn",
                        "EPOCHS": str(EPOCHS),
                        "PINN_ACTIVATION_EPOCH": str(seed_test_warmup),
                        "ALPHA_INIT": f"{seed_test_alpha:.12g}",
                        "SERIES_LABEL": f"PINN seed={seed_str} a={alpha_to_label(seed_test_alpha)} W={seed_test_warmup}",
                    },
                    color=pinn_color,
                    linestyle="-",
                    marker=marker,
                    percent_list=seed_test_percents,
                    seed_list=[seed_str],
                    alpha=seed_test_alpha,
                    warmup=seed_test_warmup,
                    group="seed",
                )
            )



    dedupe_equivalent_series_tags(series)
    apply_distinct_series_styles(series)

    # First infer and plot every completed checkpoint. Only configurations
    # without a completed checkpoint continue to the training scheduler below.
    missing_training_units = preflight_infer_plot_existing(series)

    # ---- 1. RUN EXPERIMENTS ----
    if RUN_TRAIN and not missing_training_units:
        print("[TRAIN] No missing checkpoints after preflight; skip training scheduler.")
        RUN_TRAIN = False

    if RUN_TRAIN:
        def _progress_refresh(reason=""):
            print(f"[LIVE] Trigger checkpoint scan ({reason})")
            reevaluate_existing_models_for_series(series)
            try:
                save_all_metrics_long_csv(series, REPORT_DIR)
            except Exception as exc:
                print(f"[LIVE][WARN] Metric table refresh failed: {exc}")
            if LIVE_PROGRESS_PLOTS:
                refresh_live_comparison_plots(series, REPORT_DIR)

        baseline_series = [s for s in series if s.get("group") == "baseline"]
        alpha_series = [s for s in series if s.get("group") == "alpha"]
        deferred_series = [s for s in series if s.get("group") in {"warmup", "seed"}]
        needs_auto_alpha = (
            auto_select_best_alpha
            and not best_alpha_env_set
            and "alpha" in test_groups
        )

        if LIVE_PROGRESS_PLOTS:
            refresh_live_comparison_plots(series, REPORT_DIR)

        if needs_auto_alpha:
            run_auto_alpha_pipeline(
                series=series,
                baseline_series=baseline_series,
                alpha_series=alpha_series,
                deferred_series=deferred_series,
                alpha_percent_priority=alpha_test_percents,
                update_warmup=not warmup_alpha_env_set,
                update_seed=not seed_alpha_env_set,
                progress_callback=_progress_refresh,
            )
        else:
            if auto_select_best_alpha and not best_alpha_env_set and "alpha" in test_groups:
                reevaluate_existing_models_for_series(alpha_series)
                collect_series_results(series)
                selected_alpha = choose_best_alpha_from_alpha_series(series, alpha_test_percents)
                if selected_alpha is not None:
                    best_alpha = selected_alpha
                    retarget_deferred_pinn_alpha(
                        series,
                        best_alpha,
                        update_warmup=not warmup_alpha_env_set,
                        update_seed=not seed_alpha_env_set,
                    )
                    dedupe_equivalent_series_tags(series)
                    apply_distinct_series_styles(series)
                    print(f"[AUTO] Using existing alpha sweep winner: alpha={alpha_to_label(best_alpha)}")
            run_experiment_group(series, start_index=1, progress_callback=_progress_refresh)
        # After launching/finishing training runs, ensure all seeds produced results
        def wait_for_all_seeds(series_list, max_wait_secs=60*60*6, poll_interval=30):
            """Wait until every run has a final checkpoint.

            Args:
                series_list: list of series dicts (must contain 'tag' and 'mode').
                max_wait_secs: total seconds to wait before giving up (default 6 hours).
                poll_interval: seconds between checks.
            """
            import time
            start = time.time()
            missing = True
            while True:
                all_ok = True
                missing_items = []
                for item in series_list:
                    tag = item.get('tag')
                    mode = item.get('mode')
                    percent_list = item.get('percent_list', PERCENT_LIST)
                    seed_list = item.get('seed_list', SEEDS)
                    for pct in percent_list:
                        for seed in seed_list:
                            seed_suffix = f"{tag}_seed_{seed}"
                            percent_dir = get_percent_dir(mode, pct)
                            if not os.path.isdir(percent_dir):
                                all_ok = False
                                missing_items.append((item['key'], pct, seed))
                                continue
                            # find run folders that start with the seed suffix
                            matches = [d for d in os.listdir(percent_dir) if d.startswith(seed_suffix)]
                            if not matches:
                                all_ok = False
                                missing_items.append((item['key'], pct, seed))
                                continue
                            run_dir = find_latest_run_dir(percent_dir, seed_suffix)
                            if run_dir is None:
                                all_ok = False
                                missing_items.append((item['key'], pct, seed))
                                continue
                            if not has_completed_checkpoint(run_dir):
                                all_ok = False
                                missing_items.append((item['key'], pct, seed, run_dir))

                if all_ok:
                    print(f"[OK] All seeds completed for all series/percents (checked in {time.time()-start:.1f}s)")
                    return True

                elapsed = time.time() - start
                if elapsed > max_wait_secs:
                    print(f"[WARN] Timeout waiting for seeds. Missing entries: {len(missing_items)}")
                    for m in missing_items[:10]:
                        print(f" - Missing: {m}")
                    return False

                # brief status print and sleep
                print(f"[WAIT] Waiting for {len(missing_items)} missing seed results... checking again in {poll_interval}s")
                time.sleep(poll_interval)

        wait_for_all_seeds(series)
    else:
        print("[INFO] RUN_TRAIN=0 -> skip training; only collect existing results.")
        if auto_select_best_alpha and not best_alpha_env_set and "alpha" in test_groups:
            reevaluate_existing_models_for_series([s for s in series if s.get("group") == "alpha"])
            collect_series_results(series)
            selected_alpha = choose_best_alpha_from_alpha_series(series, alpha_test_percents)
            if selected_alpha is not None:
                best_alpha = selected_alpha
                retarget_deferred_pinn_alpha(
                    series,
                    best_alpha,
                    update_warmup=not warmup_alpha_env_set,
                    update_seed=not seed_alpha_env_set,
                )
                dedupe_equivalent_series_tags(series)
                apply_distinct_series_styles(series)
                print(f"[AUTO] Using existing alpha sweep winner for reports: alpha={alpha_to_label(best_alpha)}")

    # ---- 1b. RE-EVALUATE EXISTING CHECKPOINTS WITHOUT TRAINING ----
    # If a configured run already has a model checkpoint, rebuild the same
    # train/val/test split from percent+seed, run inference on test, and refresh
    # the CSV files used by the plots below.
    reevaluate_existing_models_for_series(series)
    save_all_metrics_long_csv(series, REPORT_DIR)

    # ---- 2. GATHER RESULTS ----
    print("\n[INFO] GATHERING RESULTS & PLOTTING...")
    all_percents = set()
    rows = []

    # For each series, collect results across seeds
    all_percents = collect_series_results(series)

    target_percents = [p for p in PERCENT_LIST if p in all_percents]
    if not target_percents:
        print("[WARNING] No result data found. Check training/output folders.")
        sys.exit(0)

    # Create a flat table with one row per (percent, series, seed)
    for pct in target_percents:
        for item in series:
            seed_list = item.get('seed_list', SEEDS)
            for seed in seed_list:
                rec = item.get("per_seed_results", {}).get(seed, {}).get(pct)
                row = {
                    "train_percent": pct,
                    "series_key": item["key"],
                    "series_label": item["label"],
                    "group": item.get("group", ""),
                    "mode": item["mode"],
                    "custom_tag": f"{item['tag']}_seed_{seed}",
                    "seed": seed,
                    "rmse_avg": np.nan if rec is None else rec.get("rmse", np.nan),
                    "mae_avg": np.nan if rec is None else rec.get("mae", np.nan),
                    "nrmse_avg": np.nan if rec is None else rec.get("nrmse", np.nan),
                    "r2_avg": np.nan if rec is None else rec.get("r2", np.nan),
                    "max_error_avg": np.nan if rec is None else rec.get("max_error", np.nan),
                    "clf_acc_percent": np.nan if rec is None else rec.get("acc", np.nan),
                    "clf_balanced_acc_percent": np.nan if rec is None else rec.get("balanced_acc", np.nan),
                    "clf_f1_macro_percent": np.nan if rec is None else rec.get("f1_macro", np.nan),
                    "clf_mcc": np.nan if rec is None else rec.get("mcc", np.nan),
                }
                if rec is not None:
                    for key, value in rec.items():
                        if key in {"rmse", "mae", "nrmse", "r2", "max_error", "acc", "balanced_acc", "f1_macro", "mcc", "run_dir", "csv_file"}:
                            continue
                        row[key] = value
                rows.append(row)

    metrics_df = pd.DataFrame(rows)
    summary_csv = os.path.join(REPORT_DIR, "alpha_sweep_metrics_summary.csv")
    _atomic_to_csv(metrics_df, summary_csv, index=False)
    print(f"[OK] Saved metrics summary: {summary_csv}")

    # ---- 2b. AGGREGATE ACROSS SEEDS AND COMPUTE STATISTICS (No-PINN vs PINN) ----
    stats_rows = []
    stat_metrics = [("rmse", "RMSE (mm)", "rmse_avg"), ("mae", "MAE (mm)", "mae_avg"), ("nrmse", "NRMSE (%)", "nrmse_avg"), ("acc", "Accuracy (%)", "clf_acc_percent")]
    
    pinn_keys = [
        s['key'] for s in series
        if s['mode'] == 'pinn' and s.get("group") != "seed"
    ]
    nopinn_keys = [
        s['key'] for s in series
        if s['mode'] == 'no_pinn' and s.get("group") != "seed"
    ]

    for metric_name, metric_label, col_name in stat_metrics:
        if col_name not in metrics_df.columns:
            continue
        by_series_pct = defaultdict(lambda: defaultdict(list))
        for _, r in metrics_df.iterrows():
            key = r['series_key']
            pct = int(r['train_percent'])
            val = r.get(col_name, np.nan)
            try:
                if np.isnan(val):
                    continue
            except Exception:
                pass
            by_series_pct[key][pct].append(float(val))

        for pct in target_percents:
            for pk in pinn_keys:
                for nk in nopinn_keys:
                    pinn_vals = by_series_pct.get(pk, {}).get(pct, [])
                    nopinn_vals = by_series_pct.get(nk, {}).get(pct, [])
                    if not pinn_vals or not nopinn_vals:
                        continue
                    n = min(len(pinn_vals), len(nopinn_vals))
                    pinn_arr = np.array(pinn_vals[:n])
                    nopinn_arr = np.array(nopinn_vals[:n])

                    mean_p = float(np.mean(pinn_arr))
                    sd_p = float(np.std(pinn_arr, ddof=1)) if n > 1 else 0.0
                    ci_p = 1.96 * (sd_p / math.sqrt(n)) if n > 1 else 0.0

                    mean_n = float(np.mean(nopinn_arr))
                    sd_n = float(np.std(nopinn_arr, ddof=1)) if n > 1 else 0.0
                    ci_n = 1.96 * (sd_n / math.sqrt(n)) if n > 1 else 0.0

                    t_stat, t_p = (None, None)
                    w_stat, w_p = (None, None)
                    if n > 1:
                        try:
                            t_res = stats.ttest_rel(pinn_arr, nopinn_arr, nan_policy='omit')
                            t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
                        except Exception:
                            t_stat, t_p = (None, None)
                        try:
                            w_res = stats.wilcoxon(pinn_arr, nopinn_arr)
                            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
                        except Exception:
                            w_stat, w_p = (None, None)

                    diffs = pinn_arr - nopinn_arr
                    cohens_d_paired = float(np.mean(diffs) / np.std(diffs, ddof=1)) if n > 1 and np.std(diffs, ddof=1) > 0 else 0.0

                    stats_rows.append({
                        'percent': pct,
                        'metric': metric_name,
                        'metric_label': metric_label,
                        'pinn_key': pk,
                        'nopinn_key': nk,
                        'n_pairs': n,
                        'pinn_mean': mean_p,
                        'pinn_sd': sd_p,
                        'pinn_ci95': ci_p,
                        'nopinn_mean': mean_n,
                        'nopinn_sd': sd_n,
                        'nopinn_ci95': ci_n,
                        't_stat': t_stat,
                        't_pvalue': t_p,
                        'wilcoxon_stat': w_stat,
                        'wilcoxon_pvalue': w_p,
                        'cohens_d_paired': cohens_d_paired,
                    })

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(REPORT_DIR, 'per_percent_paired_stats.csv')
    if not stats_df.empty:
        _atomic_to_csv(stats_df, stats_csv, index=False)
        print(f"[OK] Saved per-percent paired statistics: {stats_csv}")
    else:
        print("[WARN] No paired statistics could be computed")

    # Consolidate epoch_loss_log.csv từ tất cả runs thành 1 file tổng hợp
    consolidated_csv = consolidate_loss_logs(target_percents, series, REPORT_DIR)

    print("\nCollected Values (Q1 Standards):")
    header = "Pct | " + " | ".join([f"RMSE {item['short_label']} | ACC {item['short_label']}" for item in series])
    print(header)
    for p in target_percents:
        cells = [f"{p:02d}%"]
        for item in series:
            rec = item["results"].get(p)
            if rec is None:
                cells.append("nan")
                cells.append("nan")
            else:
                cells.append(f"{rec.get('rmse', np.nan):.4f}")
                cells.append(f"{rec.get('acc', np.nan):.2f}%")
        print(" | ".join(cells))

    # ---- 3. PLOTTING (GLOBAL LINES, SPLIT SEED TEST) ----
    main_series = [s for s in series if s.get("group") not in ("seed", "warmup")]
    seed_series = [s for s in series if s.get("group") == "seed"]
    if not seed_series:
        alphas = [s for s in series if s.get("group") == "alpha"]
        if alphas:
            a1 = [a for a in alphas if a.get("alpha") == 1.0]
            seed_series.append(a1[0] if a1 else alphas[-1])
    seed_series += [s for s in series if s.get("group") == "baseline"]
    warmup_series = [s for s in series if s.get("group") == "warmup"]

    mae_plot = os.path.join(REPORT_DIR, "comparison_mae_main_groups.png")
    rmse_plot = os.path.join(REPORT_DIR, "comparison_rmse_main_groups.png")
    r2_plot = os.path.join(REPORT_DIR, "comparison_r2_main_groups.png")
    nrmse_plot = os.path.join(REPORT_DIR, "comparison_nrmse_main_groups.png")
    acc_plot = os.path.join(REPORT_DIR, "comparison_acc_main_groups.png")
    if main_series:
        main_percents = sorted({
            pct for s in main_series for pct in s.get("results", {}).keys()
        })
        if main_percents:
            main_line_plots = [
                ("mae", "Average Test MAE (mm) - main groups", "Average Test MAE (mm) $\\downarrow$", "comparison_mae_main_groups.png", True),
                ("rmse", "Average Test RMSE (mm) - main groups", "Average Test RMSE (mm) $\\downarrow$", "comparison_rmse_main_groups.png", True),
                ("r2", "Test R² Score - main groups", "Test $R^2$ Score $\\uparrow$", "comparison_r2_main_groups.png", False),
                ("max_error", "Maximum Error (mm) - main groups", "Maximum Error (mm) $\\downarrow$", "comparison_max_error_main_groups.png", True),
                ("nrmse", "Average Test NRMSE (%) - main groups", "Average Test NRMSE (%) $\\downarrow$", "comparison_nrmse_main_groups.png", True),
                ("acc", "Classification Accuracy (%) - main groups", "Accuracy (%) $\\uparrow$", "comparison_acc_main_groups.png", False),
                ("balanced_acc", "Balanced Accuracy (%) - main groups", "Balanced Accuracy (%) $\\uparrow$", "comparison_balanced_acc_main_groups.png", False),
                ("f1_macro", "Macro F1-Score (%) - main groups", "Macro F1 (%) $\\uparrow$", "comparison_f1_macro_main_groups.png", False),
                ("mcc", "Matthews Correlation Coefficient - main groups", "MCC $\\uparrow$", "comparison_mcc_main_groups.png", False),
            ]
            for metric_key, plot_title, y_lbl, filename, lower_is_better in main_line_plots:
                save_line_plot(
                    main_percents,
                    main_series,
                    metric_key=metric_key,
                    title=plot_title,
                    y_label=y_lbl,
                    out_file=os.path.join(REPORT_DIR, filename),
                    lower_is_better=lower_is_better,
                )

            # ---- 4. PLOTTING (BY PERCENT BARS, MAIN) ----
            save_percent_bar_plots(main_percents, main_series, REPORT_DIR)

    if warmup_series:
        save_warmup_metric_bar_plots(warmup_series, REPORT_DIR)
    save_pinn_effect_feature_visualizations(series, REPORT_DIR)

    # Seed test gets separate figures/folder and is never mixed into main charts.
    if seed_series:
        seed_percents = sorted({
            pct for s in seed_series for pct in s.get("results", {}).keys()
        })
        if seed_percents:
            seed_report_dir = os.path.join(REPORT_DIR, "seed_test")
            os.makedirs(seed_report_dir, exist_ok=True)
            save_seed_grouped_bar_plots(seed_percents, seed_series, seed_report_dir)

    config_txt = os.path.join(REPORT_DIR, "run_config.txt")
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write("RUN CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"percents={','.join(str(p) for p in PERCENT_LIST)}\n")
        f.write(f"batch_size={BATCH_SIZE}\n")
        f.write(f"epochs={EPOCHS}\n")
        f.write(f"pinn_activation_epoch={PINN_ACTIVATION_EPOCH}\n")
        f.write(f"pinn_loss_type={PINN_LOSS_TYPE}\n")
        f.write(f"alpha_values={','.join(alpha_to_label(a) for a in ALPHA_VALUES)}\n")
        f.write(f"test_groups={','.join(test_groups)}\n")
        f.write(f"alpha_test_percents={','.join(str(p) for p in alpha_test_percents)}\n")
        f.write(f"alpha_test_seeds={','.join(str(s) for s in alpha_test_seeds)}\n")
        f.write(f"seed_test_percents={','.join(str(p) for p in seed_test_percents)}\n")
        f.write(f"seed_test_seeds={','.join(str(s) for s in seed_test_seeds)}\n")
        f.write(f"run_train={RUN_TRAIN}\n")
        f.write(f"skip_if_exists={SKIP_IF_EXISTS}\n")
        f.write(f"reuse_completed_no_pinn={REUSE_COMPLETED_NO_PINN}\n")
        f.write(f"reeval_if_model_exists={REEVAL_IF_MODEL_EXISTS}\n")
        f.write(f"report_reeval_only={REPORT_REEVAL_ONLY}\n")

    # ---- 4b. AGGREGATE PER-SHAPE RESULTS (PINN vs No-PINN per shape x percent) ----
    print("\n" + "=" * 80)
    print("AGGREGATING PER-SHAPE RESULTS (Q1 STANDARDS)")
    print("=" * 80)
    
    try:
        per_shape_csvs = []
        for pattern in [os.path.join(OUTPUT_NO_PINN_ROOT, "**", "test_metrics_per_shape.csv"),
                       os.path.join(OUTPUT_PINN_ROOT, "**", "test_metrics_per_shape.csv")]:
            per_shape_csvs.extend(glob.glob(pattern, recursive=True))
        per_shape_csvs = filter_preferred_metric_csvs(per_shape_csvs)
        
        if per_shape_csvs:
            print(f"[INFO] Found {len(per_shape_csvs)} per-shape CSVs")
            
            # Load and combine
            per_shape_dfs = []
            for csv in per_shape_csvs:
                try:
                    df = pd.read_csv(csv)
                    # Extract mode, percent, seed from path
                    path_parts = csv.replace("\\", "/").split("/")
                    mode = "PINN" if "Outputs_pinn" in csv else "NO_PINN"
                    train_pct = None
                    seed = None
                    
                    for part in path_parts:
                        if part.startswith("train_") and part.endswith("pct"):
                            train_pct = int(part.split("_")[1].replace("pct", ""))
                        if "seed_" in part:
                            seed = int(part.split("seed_")[1].split("_")[0])
                    
                    if train_pct in PERCENT_LIST:
                        df['mode'] = mode
                        df['train_percent'] = train_pct
                        df['seed'] = seed
                        df['source_csv'] = csv
                        per_shape_dfs.append(df)
                except Exception as e:
                    pass
            
            if per_shape_dfs:
                per_shape_combined = pd.concat(per_shape_dfs, ignore_index=True)
                per_shape_csv = os.path.join(REPORT_DIR, "all_per_shape_results.csv")
                _atomic_to_csv(per_shape_combined, per_shape_csv, index=False)
                print(f"[OK] Saved combined per-shape results: {per_shape_csv}")
                
                # Summary stats per mode x shape x percent
                try:
                    agg_dict = {
                        'clf_acc': ['mean', 'std'],
                        'mae_avg': ['mean', 'std'],
                        'n_samples': 'mean'
                    }
                    if 'rmse_avg' in per_shape_combined.columns:
                        agg_dict['rmse_avg'] = ['mean', 'std']
                    if 'nrmse_avg' in per_shape_combined.columns:
                        agg_dict['nrmse_avg'] = ['mean', 'std']
                    if 'r2_avg' in per_shape_combined.columns:
                        agg_dict['r2_avg'] = ['mean', 'std']
                    summary = per_shape_combined.groupby(['mode', 'train_percent', 'shape']).agg(agg_dict).round(4)
                    summary_csv = os.path.join(REPORT_DIR, "per_shape_summary_stats.csv")
                    _atomic_to_csv(summary, summary_csv)
                    print(f"[OK] Saved per-shape summary stats: {summary_csv}")
                except Exception as e:
                    print(f"[WARN] Failed to generate summary stats: {e}")
                
                # PINN vs No-PINN comparison
                try:
                    comparison_rows = []
                    for pct in sorted(per_shape_combined['train_percent'].unique()):
                        for shape in sorted(per_shape_combined['shape'].unique()):
                            pinn_data = per_shape_combined[(per_shape_combined['mode']=='PINN') & 
                                                          (per_shape_combined['train_percent']==pct) &
                                                          (per_shape_combined['shape']==shape)]
                            nopinn_data = per_shape_combined[(per_shape_combined['mode']=='NO_PINN') &
                                                            (per_shape_combined['train_percent']==pct) &
                                                            (per_shape_combined['shape']==shape)]
                            
                            if len(pinn_data) > 0 and len(nopinn_data) > 0:
                                p_acc = pinn_data['clf_acc'].mean()
                                np_acc = nopinn_data['clf_acc'].mean()
                                p_mae = pinn_data['mae_avg'].mean()
                                np_mae = nopinn_data['mae_avg'].mean()
                                p_rmse = pinn_data['rmse_avg'].mean() if 'rmse_avg' in pinn_data.columns else np.nan
                                np_rmse = nopinn_data['rmse_avg'].mean() if 'rmse_avg' in nopinn_data.columns else np.nan
                                p_nrmse = pinn_data['nrmse_avg'].mean() if 'nrmse_avg' in pinn_data.columns else np.nan
                                np_nrmse = nopinn_data['nrmse_avg'].mean() if 'nrmse_avg' in nopinn_data.columns else np.nan

                                comparison_rows.append({
                                    'train_percent': pct,
                                    'shape': shape,
                                    'pinn_acc_mean': p_acc,
                                    'pinn_acc_std': pinn_data['clf_acc'].std(),
                                    'nopinn_acc_mean': np_acc,
                                    'nopinn_acc_std': nopinn_data['clf_acc'].std(),
                                    'acc_diff': p_acc - np_acc,
                                    'pinn_mae_mean': p_mae,
                                    'nopinn_mae_mean': np_mae,
                                    'mae_diff': p_mae - np_mae,
                                    'pinn_rmse_mean': p_rmse,
                                    'nopinn_rmse_mean': np_rmse,
                                    'rmse_diff': p_rmse - np_rmse,
                                    'pinn_nrmse_mean': p_nrmse,
                                    'nopinn_nrmse_mean': np_nrmse,
                                    'nrmse_diff': p_nrmse - np_nrmse,
                                })
                    
                    if comparison_rows:
                        comparison_df = pd.DataFrame(comparison_rows)
                        comparison_csv = os.path.join(REPORT_DIR, "pinn_vs_nopinn_per_shape.csv")
                        _atomic_to_csv(comparison_df, comparison_csv, index=False)
                        print(f"[OK] Saved PINN vs No-PINN comparison: {comparison_csv}")
                        
                        # Heatmaps
                        if 'shape' in comparison_df.columns and len(comparison_df) > 0:
                            for diff_col, hmap_title, hmap_file, cmap_name in [
                                ('rmse_diff', 'PINN vs No-PINN: RMSE Difference (mm, negative=PINN better)', 'per_shape_rmse_diff_heatmap.png', 'RdYlGn_r'),
                                ('mae_diff', 'PINN vs No-PINN: MAE Difference (mm, negative=PINN better)', 'per_shape_mae_diff_heatmap.png', 'RdYlGn_r'),
                                ('nrmse_diff', 'PINN vs No-PINN: NRMSE Difference (%, negative=PINN better)', 'per_shape_nrmse_diff_heatmap.png', 'RdYlGn_r'),
                                ('acc_diff', 'PINN vs No-PINN: Accuracy Difference (%, positive=PINN better)', 'per_shape_acc_diff_heatmap.png', 'RdYlGn'),
                            ]:
                                try:
                                    if diff_col in comparison_df.columns and comparison_df[diff_col].notna().any():
                                        pivot = comparison_df.pivot_table(index='shape', columns='train_percent', values=diff_col)
                                        if pivot is not None and not pivot.empty:
                                            plt.figure(figsize=(10, 6))
                                            sns.heatmap(pivot, annot=True, fmt='.3f' if 'rmse' in diff_col or 'mae' in diff_col else '.2f', cmap=cmap_name, center=0)
                                            plt.title(hmap_title, fontweight='bold')
                                            out_hmap = os.path.join(REPORT_DIR, hmap_file)
                                            plt.savefig(out_hmap, dpi=150, bbox_inches='tight')
                                            plt.close()
                                            print(f"[OK] Saved Heatmap: {out_hmap}")
                                except Exception as e:
                                    pass
                except Exception as e:
                    print(f"[WARN] Failed to generate comparison: {e}")
    except Exception as e:
        print(f"[WARN] Per-shape aggregation failed: {e}")

    # ---- 4c. AGGREGATE PER-CLASS AND PER-TARGET METRICS ----
    print("\n" + "=" * 80)
    print("AGGREGATING PER-CLASS AND W/L/D REGRESSION METRICS")
    print("=" * 80)
    consolidate_class_and_target_metrics(REPORT_DIR)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"[OK] Report folder: {REPORT_DIR}")
    print(f"[OK] Global R² plot (Thay thế MAPE): {r2_plot}")
    print(f"[OK] Global MAE plot:  {mae_plot}")
    print(f"[OK] Global RMSE plot: {rmse_plot}")
    print(f"[OK] Global NRMSE plot: {nrmse_plot}")
    print(f"[OK] Global ACC plot:  {acc_plot}")
    print(f"[OK] Per-percent folder: {os.path.join(REPORT_DIR, 'by_percent')}")
    print(f"[OK] Consolidated epoch loss log: {consolidated_csv}")
    print(f"[OK] Per-shape analysis: {os.path.join(REPORT_DIR, 'all_per_shape_results.csv')}")
    print(f"[OK] Per-class metrics: {os.path.join(REPORT_DIR, 'all_class_metrics.csv')}")
    print(f"[OK] W/L/D regression metrics: {os.path.join(REPORT_DIR, 'all_regression_metrics_by_target.csv')}")
    print(f"[OK] Experiment tracking: {os.path.join(SCRIPT_DIR, 'experiments')}/")
    print(f"[OK] Consolidated results: {os.path.join(SCRIPT_DIR, 'Result')}/")

    # ---- 5. OPTIONAL: AGGREGATE AND STATS (from detailed_metrics.csv) ----
    print("\n" + "=" * 80)
    print("AGGREGATING DETAILED METRICS AND COMPUTING STATISTICS")
    print("=" * 80)
    try:
        aggregate_detailed_metrics(REPORT_DIR)
    except Exception as e:
        print(f"[WARN] Failed to aggregate detailed metrics: {e}")
        
    # Organize all paper plots into dedicated subfolders
    try:
        organize_paper_plots(REPORT_DIR)
    except Exception as e:
        print(f"[WARN] Failed to organize paper plots: {e}")
    
    # ---- 6. MANUSCRIPT REPORTS ----
    try:
        generate_manuscript_reports()
    except Exception as e:
        print(f"[WARN] Failed to generate manuscript reports: {e}")

