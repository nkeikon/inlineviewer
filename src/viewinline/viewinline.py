#!/usr/bin/env python3
"""
viewinline — quick-look geospatial viewer with inline image support.

Supports:
  • Rasters (.tif, .tiff, .png, .jpg, .jpeg, .nc, .hdf)
  • Vectors (.shp, .geojson, .gpkg, .parquet, .geoparquet)
  • CSV (.csv) scatter plots and histograms

Display:
  Sends iTerm2-style inline image escape sequences. Works in terminals that support
  the iTerm2 inline image protocol (iTerm2, WezTerm, Konsole, etc.). In other
  terminals, the escape codes are ignored.
  
  Particularly useful on HPC systems and remote servers accessed via SSH — images
  render on your local terminal without X11 forwarding, VNC, or file downloads.
  
  No detection, no fallbacks. If images are not shown, it means that the terminal 
  is not compatible.
"""

import sys, os, base64, shutil, argparse
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from matplotlib import colormaps
import matplotlib as mpl
import subprocess
import re
import logging
import copy
from functools import wraps

try:
    import netCDF4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

import warnings

warnings.filterwarnings("ignore", message="More than one layer found", category=UserWarning)
warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid scale_factor or add_offset attribute", category=UserWarning)

__version__ = "0.3.1"

logger = logging.getLogger("viewinline")
logging.basicConfig(level=logging.INFO, format="%(message)s")

AVAILABLE_COLORMAPS = [
    "viridis", "inferno", "magma", "plasma",
    "cividis", "terrain", "RdYlGn", "coolwarm",
    "Spectral", "cubehelix", "tab10", "turbo"
]

# Terminals that don't natively support the iTerm2 OSC 1337 inline image
# protocol. Output for these is routed through chafa instead.
#
# Note: presence in this list does NOT mean "no images." Chafa auto-detects
# the terminal and picks the best output:
#   - kitty (xterm-kitty) → real images via kitty graphics protocol
#   - some others (e.g. foot, Ghostty) → may render real images via sixel
#     or kitty protocol depending on chafa's detection
#   - most others → Unicode block-art preview with 24-bit color
#     (Terminal.app, VS Code, GNOME Terminal, Alacritty, Warp, etc.)
# Only terminals without chafa installed see no rendering at all.

_TERMINALS_WITHOUT_IMAGES = [
    # macOS
    'Apple_Terminal',         # $TERM_PROGRAM for Terminal.app

    # kitty (renders real images via chafa → kitty graphics protocol)
    'xterm-kitty',            # $TERM in kitty

    # tmux / screen (TERM strings; $TMUX env var also signals tmux)
    'screen', 'screen-256color',
    'tmux', 'tmux-256color',

    # Editors / IDE terminals
    'vscode',                 # $TERM_PROGRAM in VS Code integrated terminal

    # Cross-platform terminals known not to support OSC 1337
    'alacritty',
    'foot',                   # supports sixel → chafa renders real images
    'ghostty', 'xterm-ghostty',
    'WarpTerminal',           # $TERM_PROGRAM in Warp
    'Hyper',                  # $TERM_PROGRAM in Hyper

    # Generic / legacy
    'unknown',
    'cygwin',
    'rxvt', 'rxvt-unicode', 'rxvt-unicode-256color',
    'st-256color',            # suckless st

    # Linux desktop terminals (most are VTE-based, no OSC 1337)
    'gnome-terminal',
    'xfce4-terminal',
    'lxterminal',
    'terminator',
    'tilix',
    'sakura',
    'terminology',
    'guake',
    'tilda',
    'deepin-terminal',
    'eterm',

    # Windows
    'putty',
    'Windows Terminal',
]

_SUPPORTED_RASTER_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nc", ".hdf", ".hdf5", ".h5")
_SUPPORTED_VECTOR_EXTS = (".shp", ".geojson", ".json", ".gpkg", ".parquet", ".geoparquet")


def quiet(func):
    """Decorator to suppress stdout and stderr during function execution.

    Useful for rendering thumbnails in gallery mode.
    
    Usage:
    x = quiet(some_function)(*args, **kwargs)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_handlers = [h for h in logger.handlers]
        for h in old_handlers:
            logger.removeHandler(h)
        try:
            return func(*args, **kwargs)
        finally:
            for h in old_handlers:
                logger.addHandler(h)
            pass
    return wrapper

def detect_terminal() -> dict[str, str]:
    """Detect terminal emulator by checking environment variables and parent process.
    Returns a dictionary of detected terminal-related variables.
    """
    env = os.environ
    terms = {}
    # Check for common terminal-specific variables
    for var in [
        "TERM_PROGRAM", "KONSOLE_VERSION", "KONSOLE_PROFILE_NAME",
        "VTE_VERSION", "TERMINATOR_UUID", "ALACRITTY_SOCKET",
        "WEZTERM_EXECUTABLE", "ITERM_SESSION_ID"
    ]:
        if var in env:
            term = env[var]
            if term:
                terms[var] = term
    # Fallback: check parent process
    if not terms:
        try:
            ppid = os.getppid()
            parent = subprocess.check_output(["ps", "-p", str(ppid), "-o", "comm="]).decode().strip()
            if parent:
                terms["PARENT_PROCESS"] = parent
        except Exception:
            pass
    term = env.get("TERM")
    if term:
        terms["TERM"] = term
    return terms

def is_terminal_without_images(term_info: dict[str, str]) -> bool:
    """Determine if the terminal is likely to support inline images based on detected info.
    Args:
        term_info: Dictionary of terminal-related environment variables.
    Returns:
        True if terminal is likely supported, False if known to be unsupported.
    """
    if os.environ.get("INLINE_VIEWER_ENGINE") == "chafa":
        return True
    terms = term_info.values()
    for unsupported in _TERMINALS_WITHOUT_IMAGES:
        if unsupported in terms:
            return True
    return False

def is_chafa_available() -> bool:
    """Check if the 'chafa' command-line tool is available for ASCII art fallback."""
    try:
        subprocess.run(["chafa", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

_TERMINAL_INFO = detect_terminal()
_TERMINAL_SUPPORTS_IMAGES = not is_terminal_without_images(_TERMINAL_INFO)

# ---------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------
def show_inline_image(image_array: np.ndarray, display_scale = None, is_vector: bool = False) -> None:
    """Encode image and write iTerm2 inline escape sequence to stdout.

    Raises only if image encoding fails. Cannot detect whether the terminal
    actually rendered the image.
    """
    buffer = BytesIO()
    # Use lower compression level for performance in speed
    Image.fromarray(image_array).save(buffer, format="PNG", compress_level=1, optimize=False)
    image_bytes = buffer.getvalue()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    if display_scale is None:
        width_pct = 33  # same default for both
    else:
        if is_vector:
            # For vectors, keep relative to base 33%
            width_pct = int(33 * display_scale)
        else:
            # For rasters, treat --display as absolute percentage
            width_pct = int(100 * display_scale)

    # Clamp range
    width_pct = max(5, min(width_pct, 400))

    if _TERMINAL_SUPPORTS_IMAGES:
        sys.stdout.write(f"\033]1337;File=inline=1;width={width_pct}%:{encoded}\a\n")
    else: 
        if is_chafa_available():
            # Inside tmux, force chafa to use block-art symbols instead of
            # graphics protocols. Tmux mangles kitty graphics and sixel
            # protocols, producing dot-character garbage on screen. Block-art
            # passes through tmux reliably on every outer terminal.
            chafa_args = ["chafa", "-"]
            if os.environ.get("TMUX"):
                chafa_args = ["chafa", "-f", "symbols", "-"]
            chafa_output = subprocess.check_output(
                chafa_args,
                input=image_bytes
            ).decode()
            
            sys.stdout.write(f"\n{chafa_output}\a\n")
            
        else:
            sys.stdout.write(f"[INFO] Use supported terminal or install 'chafa' for ascii art fallback. Detected: {_TERMINAL_INFO}\n")
    
    sys.stdout.flush()

def show_image_auto(img: np.ndarray, display_scale=None, is_vector: bool = False) -> None:
    """Render an image inline, with chafa fallback for non-iTerm2 terminals.
    
    Cascade:
      1. If terminal supports OSC 1337 → emit iTerm2 inline image sequence.
      2. Else if chafa is installed → pipe through chafa (which auto-detects
         and emits the terminal's native graphics protocol or block-art).
      3. Else → print an info message suggesting chafa installation.
    
    The branching happens inside show_inline_image(); this wrapper handles
    status messaging and exception safety.
    """
    try:
        show_inline_image(img, display_scale, is_vector)
        if _TERMINAL_SUPPORTS_IMAGES:
            logger.info("[VIEW] Inline render complete")
        elif is_chafa_available():
            logger.info("[VIEW] Inline render complete via chafa")    
        # If neither path applies, show_inline_image already printed the info message
    except Exception as e:
        logger.error(f"[ERROR] Failed to render image: {e}")
        import traceback
        trace = traceback.format_exc()
        logger.error(trace)

def resize_to_terminal(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize image to fit terminal window (approx 8x16 pixel cells)."""
    cols, rows = shutil.get_terminal_size((100, 40))
    max_w = cols * 8
    max_h = rows * 16
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    pil_img = Image.fromarray(img)
    pil_img = ImageOps.contain(pil_img, (new_w, new_h))
    return np.array(pil_img), scale

# ---------------------------------------------------------------------
# CSV handling
# ---------------------------------------------------------------------

# =============================================================
# Core DataFrame loader (used everywhere)
# =============================================================
def load_csv_to_df(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"[ERROR] Failed to read CSV: {e}")
        return pd.DataFrame()

# =============================================================
# Preview
# =============================================================
def preview_df(df, max_rows: int = 10, query_mode: bool = False, filename: str = None) -> None:
    """Preview a pandas DataFrame."""

    if df is None or df.empty:
        logger.warning("[WARN] No rows to preview.")
        return

    n_rows, n_cols = df.shape

    # -------------------------------------------------------------
    # Print dataset header (ONLY when not query mode)
    # -------------------------------------------------------------
    if not query_mode:
        name = filename if filename else "DataFrame"
        logger.info(f"[DATA] CSV file: {name} — {n_rows:,} rows × {n_cols} columns")

    # -------------------------------------------------------------
    # Decide how many rows to show
    # -------------------------------------------------------------
    if n_rows <= max_rows:
        rows_to_show = df
    else:
        ans = input(f"Preview first {max_rows} rows? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            return
        rows_to_show = df.head(max_rows)

    # -------------------------------------------------------------
    # Build pretty table
    # -------------------------------------------------------------
    columns = rows_to_show.columns.tolist()

    col_widths = []
    for col in columns:
        values = [str(col)] + rows_to_show[col].astype(str).tolist()
        width = min(max(len(v) for v in values), 22)
        col_widths.append(width)

    def fmt_row(row):
        return "| " + " | ".join(
            f"{str(val)[:w]:<{w}}" for val, w in zip(row, col_widths)
        ) + " |"

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    logger.info(sep)
    logger.info(fmt_row(columns))
    logger.info(sep)

    for _, r in rows_to_show.iterrows():
        logger.info(fmt_row(r.tolist()))

    logger.info(sep)

    logger.info("[INFO] Use --describe for summary or --hist for histograms.")

# =============================================================
# Describe
# =============================================================
def describe_df(df, column=None):
    if df is None or df.empty:
        logger.warning("[WARN] No data rows found.")
        return

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        logger.info("[INFO] No numeric columns found.")
        return

    if column:
        if column not in numeric_df.columns:
            logger.warning(f"[WARN] Column '{column}' not numeric.")
            return
        numeric_df = numeric_df[[column]]
        logger.info(f"[SUMMARY] Column '{column}' (describe):")
    else:
        logger.info("[SUMMARY] Numeric columns (describe):")

    headers = ["Column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    col_widths = [12, 8, 10, 10, 10, 10, 10, 10, 10]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    logger.info(sep)
    logger.info(fmt.format(*headers))
    logger.info(sep)

    for name in numeric_df.columns:
        vals = numeric_df[name].dropna().astype(float).values
        if len(vals) == 0:
            continue

        vals.sort()
        n = len(vals)

        row = [
            name[:12],
            n,
            f"{np.mean(vals):.3f}",
            f"{np.std(vals, ddof=1) if n>1 else 0:.3f}",
            f"{vals.min():.3f}",
            f"{np.percentile(vals,25):.3f}",
            f"{np.percentile(vals,50):.3f}",
            f"{np.percentile(vals,75):.3f}",
            f"{vals.max():.3f}",
        ]

        logger.info(fmt.format(*row))

    logger.info(sep)

# =============================================================
# Histogram
# =============================================================
def inline_histogram_df(df, column=None, bins=20, display_scale=None, is_vector=False):

    if df is None or df.empty:
        logger.warning("[WARN] No data to plot.")
        return

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        logger.info("[INFO] No numeric columns found.")
        return

    if column:
        if column not in numeric_df.columns:
            logger.warning(f"[WARN] Column '{column}' not numeric.")
            return
        cols = [(column, numeric_df[column].dropna().values)]
    else:
        cols = [(c, numeric_df[c].dropna().values) for c in numeric_df.columns]

    draw_histograms(cols, bins, display_scale, is_vector)


def draw_histograms(cols, bins, display_scale=None, is_vector=False):
    """Render histograms and display via unified show_image_auto()"""
    import math

    if not cols:
        logger.info("[INFO] No numeric columns found.")
        return

    if len(cols) == 1:
        per_row = 1
        w, h = 400, 200
    else:
        per_row = 2
        w, h = 300, 180

    margin = 40
    nrows = math.ceil(len(cols) / per_row)
    total_w = per_row * w + (per_row + 1) * margin
    total_h = nrows * h + (nrows + 1) * margin

    canvas = Image.new("RGB", (total_w, total_h), (220, 220, 220))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (col, vals) in enumerate(cols):

        vals = np.array(vals, dtype=float)
        if len(vals) == 0:
            continue

        counts, edges = np.histogram(vals, bins=bins)

        idx = np.argmax(counts)
        logger.info((
            f"[INFO] Column: {col}\n"
            f"       Most frequent range: {edges[idx]:.2f} – {edges[idx+1]:.2f}\n"
            f"       Values in this range: {counts[idx]}"))

        counts = counts.astype(float)
        counts /= counts.max() if counts.max() else 1

        row_i = i // per_row
        col_i = i % per_row
        x0 = margin + col_i * (w + margin)
        y0 = margin + row_i * (h + margin)

        # Draw column title
        title = col[:22]

        if font:
            tw, th = draw.textbbox((0,0), title, font=font)[2:]
        else:
            tw, th = draw.textsize(title)

        draw.text(
            (x0 + (w - tw) / 2, y0 - 25),
            title,
            fill=(40, 40, 40),
            font=font
        )

        bw = w / bins
        base_y = y0 + h - 25
        cmap = colormaps["viridis"]

        for j, c in enumerate(counts):
            bar_h = int(c * (h - 50))
            x1 = int(x0 + j * bw)
            x2 = int(x1 + bw - 1)
            y1 = int(base_y - bar_h)
            color = tuple(int(x * 255) for x in cmap(c)[:3])
            draw.rectangle([x1, y1, x2, base_y], fill=color)

        mn = edges[0]
        mx = edges[-1]

        def fmt(v):
            if float(v).is_integer():
                return f"{int(v)}"
            return f"{v:.2f}"

        draw.text((x0 + 2, base_y + 5), fmt(mn), fill=(90,90,90), font=font)
        draw.text((x0 + w - 55, base_y + 5), fmt(mx), fill=(90,90,90), font=font)

    # Convert to numpy array and use unified display
    img_array = np.array(canvas)
    show_image_auto(img_array, display_scale, is_vector)


# =============================================================
# Scatter
# =============================================================
def plot_scatter_df(df, x_col: str, y_col: str, display_scale=None, is_vector=False):
    import io

    if df is None or df.empty:
        logger.warning("[WARN] No data to plot.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        logger.error(f"[ERROR] Columns '{x_col}' or '{y_col}' not found.")
        return

    df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        logger.warning("[WARN] No numeric values to plot.")
        return

    w, h = 420, 300
    margin = 40

    img = Image.new("RGB", (w, h), (220, 220, 220))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x_vals = df[x_col].to_numpy()
    y_vals = df[y_col].to_numpy()

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    if x_max == x_min or y_max == y_min:
        logger.warning("[WARN] Scatter plot requires at least two distinct values.")
        return

    def scale_x(x):
        return margin + (x - x_min) / (x_max - x_min) * (w - 2 * margin)

    def scale_y(y):
        return h - margin - (y - y_min) / (y_max - y_min) * (h - 2 * margin)

    draw.line([(margin, h - margin), (w - margin, h - margin)], fill=(100,100,100))
    draw.line([(margin, h - margin), (margin, margin)], fill=(100,100,100))

    for x, y in zip(x_vals, y_vals):
        px, py = scale_x(x), scale_y(y)
        s = 1.5
        draw.line([(px - s, py), (px + s, py)], fill=(70,130,180))
        draw.line([(px, py - s), (px, py + s)], fill=(70,130,180))

    draw.text((margin + 2, h - margin + 8), x_col[:14], fill=(25,25,25), font=font)
    draw.text((6, margin - 15), y_col[:14], fill=(25,25,25), font=font)

    # Add min / max axis labels
    def fmt(v):
        if abs(v) >= 1000:
            return f"{int(v):,}"
        if float(v).is_integer():
            return f"{int(v)}"
        return f"{v:.2f}"

    draw.text((margin, h - margin + 20), fmt(x_min), fill=(90,90,90), font=font)
    draw.text((w - margin - 50, h - margin + 20), fmt(x_max), fill=(90,90,90), font=font)
    draw.text((5, h - margin - 5), fmt(y_min), fill=(90,90,90), font=font)
    draw.text((5, margin - 5), fmt(y_max), fill=(90,90,90), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    show_image_auto(np.array(Image.open(buf)), display_scale, is_vector)

# ---------------------------------------------------------------------
# Raster handling
# ---------------------------------------------------------------------
def normalize_to_uint8(band: np.ndarray, vmin=None, vmax=None, nodata=None) -> np.ndarray:
    band = band.astype(float)

    # Start with finite pixels
    valid = np.isfinite(band)

    # Apply nodata override if provided
    if nodata is not None:
        valid &= (band != nodata)

    if not np.any(valid):
        return np.zeros_like(band, dtype=np.uint8)

    valid_vals = band[valid]

    # --- Manual scaling ---
    if vmin is not None and vmax is not None:
        mn, mx = vmin, vmax
        logger.info(f"[VIEW] Using manual scaling: {mn} to {mx}")

    # --- Percentile fallback ---
    else:
        if valid_vals.size < 1_000_000:
            sample = valid_vals
        else:
            sample = np.random.choice(valid_vals, 1_000_000, replace=False)

        mn, mx = np.percentile(sample, (2, 98))

    if mx <= mn:
        return np.zeros_like(band, dtype=np.uint8)

    band = np.clip((band - mn) / (mx - mn), 0, 1)
    band[~valid] = 0

    return (band * 255).astype(np.uint8)


def render_simple_image(filepath: str, args) -> None:
    """Render a simple PNG/JPG image using PIL (no geospatial processing)."""
    try:
        img_pil = Image.open(filepath)
        img_array = np.array(img_pil)
        
        # Handle different formats
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        H, W = img_array.shape[:2]
        logger.info(f"[DATA] Image loaded: {os.path.basename(filepath)} ({W}×{H})")
        
        if args.display:
            new_w, new_h = max(1, int(W * args.display)), max(1, int(H * args.display))
            img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.BILINEAR))
            logger.info(f"[VIEW] Manual resize ×{args.display:.2f} → {new_w}×{new_h}px")
        else:
            img_array, scale = resize_to_terminal(img_array)
            logger.info(f"[VIEW] Rendered image size → {img_array.shape[1]}×{img_array.shape[0]}px (size={scale:.2f})")
        
        show_image_auto(img_array, getattr(args, "display", None), is_vector=False)
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load image: {e}")

def render_netcdf_via_netcdf4(path, args):
    """Read a NetCDF file via netCDF4 (bypassing GDAL). Handles hierarchical
    groups and hyperspectral cubes where GDAL aborts or interprets axes wrong.
    """
    if not HAS_NETCDF4:
        logger.error((
            "[ERROR] netCDF4 not installed. Install with:\n"
            "        pip install netCDF4\n"
            "        or:  pip install viewinline[netcdf]"))
        return

    try:
        nc = netCDF4.Dataset(path)
    except Exception as e:
        logger.error(f"[ERROR] Could not open NetCDF file: {e}")
        return

    # Recursively collect (path, variable) pairs across all groups
    def collect_vars(group, prefix=""):
        out = []
        for name, var in group.variables.items():
            full_name = f"{prefix}{name}"
            out.append((full_name, var))
        for sub_name, sub in group.groups.items():
            out.extend(collect_vars(sub, f"{prefix}{sub_name}/"))
        return out

    all_vars = collect_vars(nc)

    if not all_vars:
        logger.error("[ERROR] No variables found in file.")
        nc.close()
        return

    # If no --subset, list all variables and exit
    if not args.subset:
        logger.info(f"Found {len(all_vars)} variables in {os.path.basename(path)}:")
        for i, (name, var) in enumerate(all_vars, 1):
            shape_str = "x".join(str(s) for s in var.shape)
            logger.info(f"  [{i}] {name}  ({shape_str}, {var.dtype})")
        logger.info(f"\nUse --subset <N> to display a specific variable.")
        nc.close()
        return

    # Validate --subset
    if args.subset < 1 or args.subset > len(all_vars):
        logger.error(f"[ERROR] --subset must be between 1 and {len(all_vars)}")
        nc.close()
        return

    var_name, var = all_vars[args.subset - 1]
    logger.info(f"[INFO] Displaying variable {args.subset}: {var_name}")
    logger.info(f"[DATA] Shape: {var.shape}  dtype: {var.dtype}  dims: {var.dimensions}")

    # Detect dimensionality and read the right slice
    if var.ndim == 2:
        data = np.asarray(var[:, :], dtype=np.float64)
        slice_info = "2D variable"

    elif var.ndim == 3:
        spatial_dims = {'lat', 'lon', 'latitude', 'longitude', 'y', 'x'}
        
        spectral_axis = None
        
        # 1. User override via --reduce
        if args.reduce_dim is not None:
            if args.reduce_dim in var.dimensions:
                spectral_axis = list(var.dimensions).index(args.reduce_dim)
                logger.info(f"[INFO] Using user-specified --reduce '{args.reduce_dim}'")
            else:
                logger.error(f"[ERROR] --reduce '{args.reduce_dim}' is not a dimension of this variable.")
                logger.info(f"[INFO] Available dimensions: {list(var.dimensions)}")
                nc.close()
                return
        
        # 2. Standard convention: reduce along the non-spatial dim
        if spectral_axis is None:
            has_standard_spatial = any(d in spatial_dims for d in var.dimensions)
            if has_standard_spatial:
                for i, dim_name in enumerate(var.dimensions):
                    if dim_name not in spatial_dims:
                        spectral_axis = i
                        break
        
        # 3. Fallback heuristic: smallest dim is typically the band axis
        if spectral_axis is None:
            sizes = [(i, var.shape[i]) for i in range(3)]
            spectral_axis = min(sizes, key=lambda x: x[1])[0]
            logger.info(f"[INFO] Non-standard dimensions detected: {list(var.dimensions)}")
            logger.info(f"[INFO] Reducing along '{var.dimensions[spectral_axis]}' (size {var.shape[spectral_axis]}, assumed band/spectral axis)")
            logger.info(f"[INFO] If this is not correct, use --reduce DIM_NAME to override.")
        
        # Slice along chosen axis
        band_count = var.shape[spectral_axis]
        band_num = args.band if args.band is not None else 1
        band_idx = max(0, min(band_num - 1, band_count - 1))
        slicer = [slice(None)] * 3
        slicer[spectral_axis] = band_idx
        data = np.asarray(var[tuple(slicer)], dtype=np.float64)
        slice_info = f"slice along axis {spectral_axis} ({var.dimensions[spectral_axis]}), band {band_idx + 1} of {band_count}"

    else:
        logger.error(f"[ERROR] viewinline only supports 2D or 3D variables. This one is {var.ndim}D.")
        nc.close()
        return

    logger.info(f"[DATA] {slice_info}")
    # Apply fill value
    fill = getattr(var, '_FillValue', None)
    if fill is not None:
        data = np.where(data == fill, np.nan, data)
    # Flip vertically if data is stored south-to-north so north appears at top.
    # Determine which dims remain after slicing — for the 2D result, figure out
    # which axis (0 or 1) corresponds to latitude, and check that dim's coord values.
    if var.ndim == 2:
        remaining_dims = list(var.dimensions)
    elif var.ndim == 3:
        if spectral_axis is not None:
            remaining_dims = [d for i, d in enumerate(var.dimensions) if i != spectral_axis]
        else:
            remaining_dims = list(var.dimensions[1:])  # axis 0 was reduced
    else:
        remaining_dims = []
    lat_names = {'lat', 'latitude', 'y'}
    for axis_in_2d, dim_name in enumerate(remaining_dims):
        if dim_name in lat_names and dim_name in nc.variables:
            lat_vals = nc[dim_name][:]
            if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                data = np.flip(data, axis=axis_in_2d)
                logger.info(f"[INFO] Flipped along '{dim_name}' for display (data stored south-to-north).")
            break
    nc.close()

    # Normalize and display
    band_u8 = normalize_to_uint8(data, vmin=args.vmin, vmax=args.vmax,
                                 nodata=args.nodata)

    if args.colormap:
        cmap = colormaps[args.colormap]
        colored = cmap(band_u8 / 255.0)
        img = (colored[:, :, :3] * 255).astype(np.uint8)
        logger.info(f"[INFO] Applying colormap: {args.colormap}")
    else:
        img = np.stack([band_u8] * 3, axis=-1)
        logger.info("[INFO] Displaying grayscale")

    # Resize to terminal
    H, W = img.shape[:2]
    if args.display:
        new_w, new_h = max(1, int(W * args.display)), max(1, int(H * args.display))
        img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
        logger.info(f"[VIEW] Manual resize ×{args.display:.2f} → {new_w}×{new_h}px")
    # else:
    #     img, scale = resize_to_terminal(img)
    #     logger.info(f"[VIEW] Rendered image size → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")
    else:
            max_dim = 2000
            if max(img.shape[:2]) > max_dim:
                scale = max_dim / max(img.shape[:2])
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
                logger.info(f"[VIEW] Downsampled from {W}×{H}px to {new_w}×{new_h}px (scale={scale:.2f})")
                logger.info(f"[INFO] Use --display 1 for full resolution.")
            else:
            # (matches the width_pct logic in show_inline_image)
                display_pct = args.display if args.display is not None else 0.33
            
                logger.info(f"[VIEW] Rendered image size → {img.shape[1]}×{img.shape[0]}px (size={display_pct:.2f})")  
       
    show_image_auto(img, getattr(args, "display", None), is_vector=False)

def render_raster(paths: list[str], args) -> list[np.ndarray]:
    results = []
    try:
        import rasterio
        import rasterio.enums
    except ImportError:
        logger.error("[ERROR] rasterio not installed. Please install with `pip install rasterio`.")
        return results

    try:

        if len(paths) == 1:
            path = paths[0]

            if path.lower().endswith('.nc'):
                r = render_netcdf_via_netcdf4(path, args)
                results.extend(r)
                return results
            
            # Handle NetCDF/HDF with subdatasets
        
            if path.lower().endswith(('.nc', '.hdf', '.hdf5', '.h5')):
                try:
                
                    with rasterio.open(path) as src:
                        subdatasets = src.subdatasets
                    
                    # If there are subdatasets, require --subset to select one
                    if subdatasets:
                        if not args.subset:
                            file_type = "variables" if path.lower().endswith('.nc') else "datasets"
                            logger.info(f"Found {len(subdatasets)} {file_type} in {os.path.basename(path)}:")
                            for i, sub in enumerate(subdatasets, 1):
                                # Extract dataset/variable name from GDAL subdataset string
                                ds_name = sub.split(':')[-1].lstrip('/')
                                logger.info(f"  [{i}] {ds_name}")
                            logger.info(f"\nUse --subset <N> to display a specific {file_type[:-1]}.")
                            return results
                        
                        # Select by index
                        if args.subset < 1 or args.subset > len(subdatasets):
                            logger.error(f"[ERROR] --subset must be between 1 and {len(subdatasets)}")
                            return results
                        
                        path = subdatasets[args.subset - 1]
                        var_name = path.split(':')[-1]
                        logger.info(f"[INFO] Displaying variable {args.subset}: {var_name}")
                
                except rasterio.errors.RasterioIOError as e:
                    # GDAL lacks support, try h5py fallback for HDF5
                    if path.lower().endswith(('.hdf5', '.h5')):
                        try:
                            import h5py
                        except ImportError:
                            logger.error("[ERROR] HDF5 file cannot be opened.")
                            logger.info(("[INFO] Requires either:\n"
                                         "        - GDAL with HDF5 support, or\n"
                                         "        - h5py: pip install h5py"))
                            return results
                        
                        # logger.error("[ERROR] h5py fallback not yet implemented.")
                        logger.info("[INFO] Install GDAL with HDF5")
                        return results
                    
                    elif path.lower().endswith('.hdf'):
                        logger.error(f"[ERROR] Cannot open HDF4 file: {e}")
                        logger.info("[INFO] HDF4 requires GDAL with HDF4 support")
                        return results
                    else:
                        # NetCDF error
                        logger.error(f"[ERROR] Cannot open NetCDF file: {e}")
                        return results

            # Continue with normal raster opening
            with rasterio.open(path) as ds:
                H, W = ds.height, ds.width
                logger.info(f"[DATA] Raster loaded: {os.path.basename(paths[0])} ({W}×{H})")
                band_count = ds.count
                
                # Auto-detect nodata from file metadata
                if args.nodata is None and ds.nodata is not None:
                    args.nodata = ds.nodata
                
                # Downsample for performance
                max_dim = 2000

                if max(H, W) > max_dim:
                    scale = max_dim / max(H, W)
                    out_h = int(H * scale)
                    out_w = int(W * scale)

                    data = ds.read(
                        out_shape=(band_count, out_h, out_w),
                        resampling=rasterio.enums.Resampling.bilinear
                    )

                    logger.info(f"[VIEW] Downsampled for preview → {out_w}×{out_h}px (scale={scale:.3f})")
                else:
                    data = ds.read()

            # Print band/slice count for all multi-band files
            if band_count > 1:
                if paths[0].lower().endswith('.nc'):
                    logger.info(f"[INFO] {band_count} slices detected")
                else:
                    logger.info(f"[INFO] Multi-band raster detected ({band_count} bands)")

            # MULTI BAND RGB (skip for NetCDF - treat as slices/timesteps, not RGB)
            # if band_count >= 3 and not paths[0].lower().endswith('.nc'):
            # Auto-composite to RGB only when user didn't explicitly ask for a single band
            # user_specified_band = args.band is not None and args.band != 1
            user_specified_band = args.band is not None
            if band_count >= 3 and not paths[0].lower().endswith('.nc') and not user_specified_band:

                if getattr(args, "rgb", None):
                    try:
                        rgb_idx = [b - 1 for b in args.rgb]
                        if len(rgb_idx) != 3:
                            raise ValueError
                        logger.info(f"[INFO] Using RGB bands: {args.rgb}")
                    except Exception:
                        logger.warning("[WARN] Invalid --rgb. Using default 1 2 3")
                        rgb_idx = [0, 1, 2]
                else:
                    rgb_idx = [0, 1, 2]

                rgb = np.stack([
                    normalize_to_uint8(
                        data[i],
                        vmin=args.vmin,
                        vmax=args.vmax,
                        nodata=args.nodata
                    )
                    for i in rgb_idx
                ], axis=-1)

                img = rgb

            # SINGLE BAND
            else:

                band_num = args.band if args.band is not None else 1
                band_idx = max(0, min(band_num - 1, band_count - 1))
                # logger.info(f"[INFO] Displaying band {band_idx + 1} of {band_count}")
                raw_band = data[band_idx].astype(float)

                # mask nodata
                nodata_val = args.nodata if args.nodata is not None else None
                if nodata_val is not None:
                    raw_band[raw_band == nodata_val] = np.nan

                # Compute preview range
                if np.any(np.isfinite(raw_band)):
                    min_val = np.nanmin(raw_band)
                    max_val = np.nanmax(raw_band)

                    if paths[0].lower().endswith('.nc'):
                        logger.info(f"[DATA] Slice {band_idx + 1} of {band_count} — value range: {min_val:.3f} → {max_val:.3f}")
                        if band_count > 1:
                            logger.info(f"[INFO] Use --band <N> to display a different slice")
                    else:
                        logger.info(f"[DATA] Band {band_idx + 1} of {band_count} — value range: {min_val:.3f} → {max_val:.3f}")
                                        
                else:
                    logger.warning("[WARN] No valid pixels found.")

                band = normalize_to_uint8(
                    raw_band,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    nodata=args.nodata
                )

                if args.colormap:
                    cmap = colormaps[args.colormap]
                    colored = cmap(band / 255.0)
                    img = (colored[:, :, :3] * 255).astype(np.uint8)
                    logger.info(f"[INFO] Applying colormap: {args.colormap}")
                else:
                    img = np.stack([band] * 3, axis=-1)
                    logger.info("[INFO] Displaying grayscale")

        elif len(paths) == 3:
            bands = []
            for p in paths:
                with rasterio.open(p) as ds:
                    bands.append(ds.read(1))
            shapes = {b.shape for b in bands}
            if len(shapes) != 1:
                logger.error("[ERROR] Raster sizes do not match.")
                return
            data = np.stack(bands, axis=0)
            H, W = data.shape[1:]
            logger.info(f"[DATA] RGB raster stack loaded: {W}×{H}")
            img = np.stack([normalize_to_uint8(b) for b in data], axis=-1)
            logger.info("[INFO] Displaying 3-band RGB composite")

        else:
            logger.error("[ERROR] Provide one raster or exactly three rasters for RGB.")
            return results

        # Resize for terminal
        H, W = img.shape[:2]
        if args.display:
            new_w, new_h = max(1, int(W * args.display)), max(1, int(H * args.display))
            img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
            logger.info(f"[VIEW] Manual resize ×{args.display:.2f} → {new_w}×{new_h}px")
        else:
            img, scale = resize_to_terminal(img)
            logger.info(f"[VIEW] Rendered image size → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")
        # Use unified display
        results.append(img)
        show_image_auto(img, getattr(args, "display", None), is_vector=False)

    except Exception as e:
        if paths[0].lower().endswith('.nc'):
            logger.error(f"[ERROR] Cannot display this variable.")
            logger.info("[INFO] viewinline only supports 2D or 3D NetCDF variables")
        else:
            logger.error(f"[ERROR] Inline render failed: {e}")
    finally:
        return results

class GalleryCurator:
    def __init__(self, thumb_size=(128, 128), label_mode: str = "name"):
        """A utility class to create thumbnails and labels for a gallery view.
        This class mainly just groups together some logic to make the code more organized.
        
        Args:
            thumb_size: max size for thumbnails (width, height)
            label_mode: "name" (filename only), "path" (full path), or "none"
        """
        self.thumb_size = thumb_size
        self.label_mode = label_mode

    def make_blank_thumbnail(self, f, rgb=(200, 200, 200), text='no preview') -> Image.Image:
        """Render a blank thumbnail with filename"""
        img = Image.new("RGB", self.thumb_size, rgb)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
            ext = os.path.splitext(os.path.basename(f))[-1].lower()
            text = f"*{ext}\n{text}"
            text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]
            text_x = (self.thumb_size[0] - text_w) / 2
            text_y = (self.thumb_size[1] - text_h) / 2
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        except Exception as e:
            pass
        img.thumbnail(self.thumb_size)
        return img

    def make_blank_thumbnail_and_label(self, f: str) -> tuple[Image.Image, str]:
        thumbnail = self.make_blank_thumbnail(f)
        label = self.make_label(f, format="muted")
        return thumbnail, label

    def make_error_thumbnail(self, f: str, e: Exception) -> Image.Image:
        return self.make_blank_thumbnail(f, rgb=(255, 235, 233), text=type(e).__name__)
    
    def make_plain_thumbnail(self, f: str) -> Image.Image:
        thumbnail = Image.open(f).convert("RGB")
        thumbnail.thumbnail(self.thumb_size)
        return thumbnail

    def make_tiff_thumbnail(self, f: str) -> Image.Image:
        """Render a thumbnail for a TIFF file, using rasterio to read and PIL to create the thumbnail."""
        import rasterio
        with rasterio.open(f) as ds:
            arr = ds.read()
            if arr.shape[0] >= 3:
                rgb = np.stack([normalize_to_uint8(arr[i]) for i in range(3)], axis=-1)
            else:
                band = normalize_to_uint8(arr[0])
                rgb = np.stack([band]*3, axis=-1)
            img = Image.fromarray(rgb)
            img.thumbnail(self.thumb_size)
            return img

    def make_array_thumbnail(self, arr: np.ndarray) -> Image.Image:
        """Render a thumbnail from an RGB array."""
        img = Image.fromarray(arr)
        img.thumbnail(self.thumb_size)
        return img
    
    def make_label(self, f: str, index: int | None = None, format=None) -> str:
        """Create a label for a file, optionally including an index for subdatasets/slices.
        Args:
            f: file path
            index: optional index for subdatasets/slices
            format: optional format for the label, e.g. "muted" for dimmer, "bold" for bolder text.
        """
        label = None
        if self.label_mode == "name":
            label = os.path.basename(f)
        elif self.label_mode == "path":
            label = f
        else:
            pass
        
        if label is not None and os.path.isdir(f):
            label = f"{label}{os.path.sep}"
            label = f"\033[7m{label}\033[0m" # make directory labels reversed to distinguish them visually
        
        if label is not None:
            
            if index is not None:
                label = f"{label}[{index}]"
        
            if format == "muted":
                label = f"\033[90m{label}\033[0m"
            elif format == "bold":
                label = f"\033[1m{label}\033[0m"
            else:
                pass
        
        return label

def render_gallery(folder: str, args, is_vector=False) -> None:
    """Render a folder of rasters/images as small thumbnails in a grid."""
    include = args.include
    exclude = args.exclude
    grid = getattr(args, "gallery", "4x4")
    display_scale = args.display
    thumb_size = getattr(args, "thumb_size", (128, 128))
    keep_unsupported = getattr(args, "gallery_keep_unsupported", False)
    curator = GalleryCurator(thumb_size=thumb_size, label_mode=args.gallery_labels)
    skip_n = args.gallery_skip
    
    import math
    try:
        # Parse grid
        try:
            cols, rows = map(int, grid.lower().split("x"))
        except Exception:
            cols, rows = 4, 4
        nmax = cols * rows

        # Collect files
        
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
        if not files:
            logger.warning(f"[WARN] No image/raster files found in {folder}")
            return

        if include:
            rex = re.compile(include)
            files = [f for f in files if rex.search(f)]
        if exclude:
            rex = re.compile(exclude)
            files = [f for f in files if not rex.search(f)]

        if skip_n > 0:
            files = files[skip_n:]
        elif skip_n < 0:
            files = files[:skip_n]
        else:
            pass

        # Load thumbnails
        thumbs = []
        labels = []
        for f in files:
            if len(thumbs) >= nmax:
                logger.info(f"[INFO] Gallery cannot render more than {nmax} images (adjust with --gallery).")
                break
            try:
                args_copy = copy.deepcopy(args) # to avoid mutating the original args
                ext = os.path.splitext(f)[1].lower()
                if ext in _SUPPORTED_RASTER_EXTS:
                    if ext in [".tif", ".tiff"]:
                        thumb = curator.make_tiff_thumbnail(f)
                        thumbs.append(thumb)
                        labels.append(curator.make_label(f))
                    elif ext in (".png", ".jpg", ".jpeg",):
                        thumb = curator.make_plain_thumbnail(f)
                        thumbs.append(thumb)
                        labels.append(curator.make_label(f))
                    elif ext in (".nc", ".hdf", ".hdf5", ".h5"):
                        arrays = render_raster([f], args_copy)
                        if not arrays:
                            thumb, label = curator.make_blank_thumbnail_and_label(f)
                            thumbs.append(thumb)
                            labels.append(label)
                            continue
                        for arr_i, arr in enumerate(arrays):
                            thumb = curator.make_array_thumbnail(arr)
                            thumbs.append(thumb)
                            label = curator.make_label(f, arr_i)
                            labels.append(f"{label}")
                    else:
                        thumb, label = curator.make_blank_thumbnail_and_label(f)
                        thumbs.append(thumb)
                        labels.append(label)
                elif ext in _SUPPORTED_VECTOR_EXTS:
                    arrays = render_vector(f, args_copy)
                    if not arrays:
                        thumb, label = curator.make_blank_thumbnail_and_label(f)
                        thumbs.append(thumb)
                        labels.append(label)
                        continue
                    for arr_i, arr in enumerate(arrays):
                        thumb = curator.make_array_thumbnail(arr)
                        thumbs.append(thumb)
                        label = curator.make_label(f, arr_i)
                        labels.append(f"{label}")
                else:
                    if keep_unsupported:
                        thumb, label = curator.make_blank_thumbnail_and_label(f)
                        thumbs.append(thumb)
                        labels.append(label)
                    else:
                        logger.warning(f"[WARN] Skipped {f} (unsupported file type)")

            except Exception as e:
                if keep_unsupported:
                    thumb = curator.make_error_thumbnail(f, e)
                    thumbs.append(thumb)
                    labels.append(curator.make_label(f, format="muted"))
                logger.warning(f"[WARN] Skipped {f} ({e})")

        if not thumbs:
            logger.warning("[WARN] No valid images loaded.")
            return

        # Create grid canvas
        n = len(thumbs)
        cols = min(cols, n)
        rows = math.ceil(n / cols)
        w, h = thumb_size
        margin = 8
        canvas_w = cols * w + (cols + 1) * margin
        canvas_h = h + margin
        logger.info(f"[INFO] Displaying {n} images ({cols}×{rows} grid)")
        for row_index in range(rows):
            canvas = Image.new("RGB", (canvas_w, canvas_h), (220, 220, 220))
            row_thumbs = thumbs[row_index * cols:(row_index + 1) * cols]
            row_labels = labels[row_index * cols:(row_index + 1) * cols]
            if any((True if label is not None else False for label in row_labels)):
                logger.info("[INFO] " + " | ".join(row_labels))
            for i, img in enumerate(row_thumbs):
                r, c = divmod(i, cols)
                x = margin + c * (w + margin)
                y = margin #+ r * (h + margin)
                canvas.paste(img, (x, y))
            quiet(show_image_auto)(np.array(canvas), display_scale, is_vector)

    except Exception as e:
        logger.error(f"[ERROR] Failed to render gallery: {e}")

# ---------------------------------------------------------------------
# Vector handling
# ---------------------------------------------------------------------
def render_vector(path, args) -> list[np.ndarray]:
    results = []
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from pyogrio import list_layers
    except ImportError as e:
        logger.error((
            "[ERROR] Missing dependency. Install with:\n"
            "  pip install geopandas matplotlib pyogrio"))
        return

    try:
        # Handle layer selection
        if path.lower().endswith((".shp", ".geojson", ".json", ".parquet", ".geoparquet")):
            layers = [(os.path.splitext(os.path.basename(path))[0], None)]
        else:
            layers = list_layers(path)
        if len(layers) > 1 and not getattr(args, "layer", None):
            logger.info(f"[INFO] Multiple layers found in '{os.path.basename(path)}':")
            for i, lyr in enumerate(layers, 1):
                name = lyr[0]
                geom = lyr[1] if len(lyr) > 1 and lyr[1] else "Unknown"
                logger.info(f"   {i}. {name} ({geom})")
            first = layers[0][0]
            logger.info(f"[INFO] Defaulting to first layer: '{first}' (use --layer <name> to select another).")
            args.layer = first
    except Exception as e:
        logger.warning(f"[WARN] Could not list layers: {e}")
        return results

    try:
        # Use read_parquet for parquet/geoparquet files
        if path.lower().endswith(('.parquet', '.geoparquet')):
            gdf = gpd.read_parquet(path)
        else:
            gdf = gpd.read_file(path, layer=getattr(args, "layer", None))
        logger.info(f"[DATA] Vector loaded: {os.path.basename(path)} ({len(gdf)} features)")
    except ImportError:
        logger.error("[ERROR] Parquet/GeoParquet support requires pyarrow. Install with: pip install pyarrow")
        return results
    except Exception as e:
        logger.error(f"[ERROR] Failed to read vector: {e}")
        return results
        
    # Detect non-geometry columns
    all_cols = [c for c in gdf.columns if c != gdf.geometry.name]

    if all_cols:
        n = len(all_cols)
        if not args.color_by:  # Only show columns if user didn't specify one
            logger.info(f"[INFO] Available columns ({n}):")
            if n <= 20:
                for c in all_cols:
                    logger.info(f"  {c}")
            else:
                ncols = 2 if n <= 30 else 3 if n <= 100 else 4
                nrows = (n + ncols - 1) // ncols
                padded = all_cols + [""] * (nrows * ncols - n)
                col_width = max(len(c) for c in all_cols) + 3
                for i in range(nrows):
                    row = ""
                    for j in range(ncols):
                        row += padded[i + j * nrows].ljust(col_width)
                    logger.info("  " + row.rstrip())
            logger.info("[INFO] Showing border-only view (use --color-by <column> to color features).")
    else:
        logger.info("[INFO] No attribute columns found.")
        
    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor="gray")
    ax.set_facecolor("gray")
    ax.set_axis_off()

    # Determine colormap
    column = args.color_by if args.color_by in gdf.columns else None

    if args.color_by and args.color_by not in gdf.columns:
        logger.warning(f"[WARN] Column '{args.color_by}' not found. Showing border-only view.")
        column = None

    if column and args.colormap is None:
        args.colormap = "terrain"
        logger.info("[INFO] Applying default colormap: terrain")

    cmap = colormaps.get(args.colormap) if args.colormap else None

    # Plot
    try:
        if column:
            # NUMERIC COLUMN
            if np.issubdtype(gdf[column].dtype, np.number):
                vmin, vmax = np.percentile(gdf[column].dropna(), (2, 98))
                logger.info(f"[INFO] Coloring by numeric column '{column}' (range: {vmin:.2f}–{vmax:.2f})")

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = colormaps.get(args.colormap or "terrain")

                geom_type = gdf.geom_type.iloc[0]

                if geom_type.startswith("Line"):
                    for _, row in gdf.iterrows():
                        val = row[column]
                        color = cmap(norm(val))
                        ax.plot(*row.geometry.xy, color=color,
                                linewidth=getattr(args, "width", 0.7))
                else:
                    gdf.plot(
                        ax=ax,
                        column=column,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        linewidth=0,
                        edgecolor="none",
                        # linewidth=getattr(args, "width", 0.2),
                        # edgecolor=args.edgecolor,
                        zorder=1
                    )

            # CATEGORICAL COLUMN
            else:
                categories = gdf[column].dropna().unique()
                logger.info(f"[INFO] Coloring by categorical column '{column}' ({len(categories)} classes)")

                cmap = colormaps.get(args.colormap or "tab10")
                colors = [cmap(i / max(1, len(categories) - 1))
                        for i in range(len(categories))]

                for cat, color in zip(categories, colors):
                    subset = gdf[gdf[column] == cat]
                    subset.plot(
                        ax=ax,
                        facecolor=color,
                        linewidth=0,
                        edgecolor="none",                       
                        # edgecolor=args.edgecolor,
                        # linewidth=getattr(args, "width", 0.2),
                        zorder=1
                    )

        # NO COLUMN → OUTLINE ONLY
        else:
            gdf.plot(
                ax=ax,
                facecolor="none",
                edgecolor=args.edgecolor,
                linewidth=getattr(args, "width", 0.7),
                zorder=1
            )

    except Exception as e:
        logger.warning(f"[WARN] Plotting failed ({e}) — fallback to border-only.")
        gdf.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.5)

    # Save to buffer
    render_dpi = 400
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=render_dpi,
                bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))

    # Resize for display
    if getattr(args, "display", None):
        scale = float(args.display)
        new_w = max(1, int(img.shape[1] * scale))
        new_h = max(1, int(img.shape[0] * scale))
        img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
        logger.info(f"[VIEW] Manual resize ×{scale:.2f} → {new_w}×{new_h}px")
    else:
        img, scale = resize_to_terminal(img)
        logger.info(f"[VIEW] Rendered image size → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")

    # Use unified display
    results.append(img)
    show_image_auto(img, getattr(args, "display", None), is_vector=True)

    return results

# ---------------------------------------------------------------------
# Smart help formatter
# ---------------------------------------------------------------------
import argparse

class SmartDefaults(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults only when meaningful (not None or SUPPRESS)."""
    def _get_help_string(self, action):
        if action.help and "%(default)" in action.help:
            return action.help
        if action.default is None or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)

# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------
def handle_tabular_data(df: pd.DataFrame, args, filepath: str) -> None:
    """Handle CSV/parquet/vector-as-table with all tabular operations."""
    if args.sql and (args.where or args.sort or args.limit or args.select):
        logger.error("[ERROR] --sql cannot be combined with --where/--sort/--limit/--select.")
        sys.exit(1)

    if args.sql:
        try:
            import duckdb
        except ImportError:
            logger.error("[ERROR] --sql requires DuckDB. Install with: pip install duckdb")
            sys.exit(1)

        logger.info("[INFO] Executing SQL query...")

        try:
            query = args.sql.replace("data", f"read_csv_auto('{filepath}')")
            con = duckdb.connect()
            df = con.execute(query).df()
            con.close()
        except Exception as e:
            logger.error(f"[ERROR] DuckDB SQL failed: {e}")
            sys.exit(1)

        if df.empty:
            logger.warning("[WARN] Query returned no rows.")
            return

        if args.describe:
            if isinstance(args.describe, str):
                describe_df(df, column=args.describe)
            else:
                describe_df(df)
            return

        if args.hist:
            if isinstance(args.hist, str):
                inline_histogram_df(df, column=args.hist, bins=args.bins, 
                                  display_scale=args.display, is_vector=False)
            else:
                inline_histogram_df(df, bins=args.bins, 
                                  display_scale=args.display, is_vector=False)
            return

        if args.scatter:
            plot_scatter_df(df, args.scatter[0], args.scatter[1], 
                          display_scale=args.display, is_vector=False)
            return

        preview_df(df, max_rows=10, query_mode=True)
        return

    if args.where or args.sort or args.limit or args.select:
        try:
            import duckdb
        except ImportError:
            logger.error("[ERROR] Filtering requires DuckDB. Install with: pip install duckdb")
            sys.exit(1)

        logger.info("[INFO] Building query...")

        base_query = "SELECT * FROM df"

        if args.select:
            selected = ", ".join(args.select)
            logger.info(f"[INFO] Selecting columns: {selected}")
            base_query = f"SELECT {selected} FROM df"

        clauses = []

        if args.where:
            logger.info(f"[INFO] Applying filter: {args.where}")
            clauses.append(f"WHERE {args.where}")

        if args.sort:
            direction = "DESC" if args.desc else "ASC"
            logger.info(f"[INFO] Sorting by: {args.sort} ({direction})")
            clauses.append(f"ORDER BY {args.sort} {direction}")

        if args.limit:
            logger.info(f"[INFO] Limiting rows: {args.limit}")
            clauses.append(f"LIMIT {args.limit}")

        query = " ".join([base_query] + clauses)

        try:
            df = duckdb.query(query).to_df()
        except Exception as e:
            logger.error(f"[ERROR] DuckDB query failed: {e}")
            sys.exit(1)

        if df.empty:
            logger.warning("[WARN] Query returned no rows.")
            return

    if args.unique:
        col = args.unique

        if col not in df.columns:
            logger.error(f"[ERROR] Column '{col}' not found.")
            return

        vals = sorted(df[col].dropna().astype(str).unique())
        n = len(vals)

        logger.info(f"[DATA] Unique values in '{col}' ({n}):")

        if n == 0:
            logger.info("  (none)")
            return

        if n <= 10:
            for v in vals:
                logger.info(f"  {v}")
        else:
            ncols = 2 if n <= 30 else 3 if n <= 100 else 4
            nrows = (n + ncols - 1) // ncols
            vals += [""] * (nrows * ncols - n)
            col_width = max(len(v) for v in vals) + 3

            for i in range(nrows):
                row = ""
                for j in range(ncols):
                    row += vals[i + j * nrows].ljust(col_width)
                logger.info("  " + row.rstrip())

            return
    
    if args.describe:
        if isinstance(args.describe, str):
            describe_df(df, column=args.describe)
        else:
            describe_df(df)
        return

    if args.hist:
        if isinstance(args.hist, str):
            inline_histogram_df(df, column=args.hist, bins=args.bins, 
                              display_scale=args.display, is_vector=False)
        else:
            inline_histogram_df(df, bins=args.bins, 
                              display_scale=args.display, is_vector=False)
        return

    if args.scatter:
        plot_scatter_df(df, args.scatter[0], args.scatter[1], 
                      display_scale=args.display, is_vector=False)
        return

    preview_df(df, max_rows=args.limit or 10, query_mode=bool(args.where or args.sort or args.select))

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="viewinline",
        description=(
            "Quick-look geospatial viewer.\n\n"
            "Supports rasters (.tif, .tiff, .png, .jpg, .jpeg), "
            "vectors (.shp, .geojson, .gpkg), and CSV preview.\n"
            "Sends iTerm2 inline image protocol — visible in compatible terminals."
        ),
        formatter_class=SmartDefaults
    )

    # File input
    parser.add_argument(
        "paths", nargs="*",  # Zero or more (optional)
        help="Path to raster(s), vector, or CSV file. Provide 1 file or exactly 3 rasters for RGB (R G B)."
    )
    # Display options
    parser.add_argument(
        "--display", type=float, default=None,
        help="Resize only the displayed image (0.5=smaller, 2=bigger). Default: auto-fit to terminal."
    )

    # Raster options
    parser.add_argument(
        "--band", type=int, default=None,
        help="Band number to display (single raster case), or slice number for NetCDF."
    )
    parser.add_argument(
        "--timestep", type=int, default=None,
        help="Alias for --band when working with NetCDF files (1-based index)."
    )
    parser.add_argument(
        "--colormap", nargs="?", const="terrain",
        choices=AVAILABLE_COLORMAPS, default=None,
        help="Apply colormap to single-band rasters or vector coloring. Flag without value → 'terrain'."
    )
    parser.add_argument(
        "--rgb", nargs=3, type=int, metavar=('R', 'G', 'B'), default=None,
        help="Three band numbers for RGB display (e.g., --rgb 4 3 2). Overrides default 1 2 3."
    )
    parser.add_argument(
        "--rgbfiles", nargs=3, type=str, metavar=('R', 'G', 'B'),
        help="Three single-band rasters for RGB composite (e.g., --rgbfiles R.tif G.tif B.tif). Can also provide as positional arguments without the flag."
    )
    parser.add_argument(
        "--vmin", type=float, default=None,
        help="Minimum pixel value for raster display scaling."
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Maximum pixel value for raster display scaling."
    )
    parser.add_argument(
        "--nodata", type=float, default=None,
        help="Override nodata value for rasters if dataset metadata is missing or incorrect."
    )
    parser.add_argument(
        "--gallery", nargs="?", const="4x4", metavar="GRID",
        help="Display all PNG/JPG/TIF images in a folder as thumbnails (e.g., 5x5 grid)."
    )

    parser.add_argument(
        "--gallery-labels", choices=["none", "name", "path"], default="name",
        help="How to display labels in the gallery view: 'none' for no labels, 'name' for file names, 'path' for full paths."
    )

    parser.add_argument(
        "--gallery-keep-unsupported", action="store_true", default=False,
        help="In gallery mode, list unsupported files with a placeholder image instead of skipping them."
    )

    parser.add_argument(
        "--gallery-skip", type=int, default=0,
        help=(
            "Number of files to skip before starting the gallery. "
            "Applied after any other filters. "
            "Allows for paging through a large number of images in a folder "
            "(e.g. --gallery 4x4 --gallery-skip 16 to see the next 'page' of 16 images after the first 16). "
            "Note that number of files in the folder may not match number of images in the gallery "
            "(e.g. NetCDF variables with multiple slices or multi-layer vector files that may result in multiple images per file). "
            "Negative values will skip from the end instead of the beginning "
            "(e.g. --gallery-skip -16 to show only the last 16 images in a folder)."
            )
    )

    parser.add_argument(
        "--subset", type=int, default=None,
        help="Variable index for NetCDF files (e.g. --subset 1)."
    )
    parser.add_argument(
        "--reduce", dest="reduce_dim", type=str, default=None,
        metavar="DIM_NAME",
        help="For 3D NetCDF variables, specify which dimension to use as the band axis (auto-detected if omitted)."
    )

    # CSV options
    parser.add_argument(
    "--hist",
    nargs="?",
    const=True,
    help="Show histograms for all numeric columns or specify one column name."
    )
    parser.add_argument(
        "--describe",
        nargs="?",
        const=True,
        help="Show summary statistics for all numeric columns or specify one column name."
    )
    parser.add_argument(
    "--bins", type=int, default=20,
    help="Number of bins for CSV histograms (used with --hist)."
    )
    parser.add_argument(
    "--scatter", nargs=2, metavar=("X", "Y"),
    help="Plot scatter of two numeric CSV columns (e.g. --scatter area_km2 year)."
    )
    parser.add_argument(
    "--unique",
    metavar="COLUMN",
    help="Show unique values for a categorical column and exit"
    )
    parser.add_argument(
        "--where",
        type=str,
        default=None,
        help="Filter rows using SQL WHERE clause (DuckDB required). Example: --where \"year > 2010\""
    )
    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help="Sort rows by values in the specified column, ascending by default (e.g. --sort population). Use --desc to reverse."
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows shown (e.g. --limit 100)."
    )
    parser.add_argument(
        "--select",
        nargs="+",
        help="Select specific columns (space separated) (e.g. --select Country City)"
    )
    parser.add_argument(
        "--sql",
        type=str,
        help="Execute full DuckDB SQL query against CSV (advanced mode)."
    )

    # Vector options
    parser.add_argument(
        "--color-by", type=str, default=None,
        help="Numeric column to color vector features by (optional)."
    )
    parser.add_argument(
    "--width", type=float, default=0.7,
    help="Line width for vector boundaries"
    )
    parser.add_argument(
        "--edgecolor", type=str, default="#F6FF00",
        help="Edge color for vector outlines (hex or named color)."
    )
    parser.add_argument(
        "--layer", type=str, default=None,
        # help="Layer name for GeoPackage or multi-layer files."
        help="Layer name for GeoPackage/multi-layer files, or variable name for NetCDF files."
    )
    parser.add_argument(
    "--table", action="store_true",
    help="Display vector/parquet file as tabular data instead of rendering geometry."
    ) 

    parser.add_argument(
        "--include", type=str, default=None,
        help="For batch modes (e.g. gallery), only include files matching this Python regex pattern. Applied before the exclude pattern. (e.g. --include \"^2023.*\\\\.tif$\")"
    )

    parser.add_argument(
        "--exclude", type=str, default=None,
        help="For batch modes (e.g. gallery), exclude files matching this Python regex pattern. Applied after the include pattern. (e.g. --exclude \"^2022.*\\\\.tif$\")"
    )

    parser.add_argument(
        "--loglevel", choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"], default="INFO",
        help="Set logging level (default: INFO), (e.g. --loglevel ERROR)."
    )

    parser.add_argument(
        "--thumb-size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=(128, 128),
        help="Thumbnail size for gallery mode (e.g. --thumb-size 100 100)."
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.loglevel))

    # Handle --rgbfiles flag (takes precedence)
    if args.rgbfiles:
        args.paths = args.rgbfiles

    # Handle aliases
    if args.timestep is not None:
        args.band = args.timestep

    # Validate input
    if not args.paths:
        parser.error("No input file(s) provided")
    
    # Basic argument sanity check
    for bad in ("color-by", "edgecolor", "colormap", "band", "display"):
        for a in args.paths:
            if a == bad:
                logger.error((
                    f"[ERROR] Missing '--' before '{bad}'.\n"
                    "        Example:  --color-by column_name"))
                sys.exit(1)

    paths = args.paths

    if len(paths) == 1:
        p = paths[0].lower()

        if p.endswith(_SUPPORTED_RASTER_EXTS):
            # For PNG/JPG, try PIL first (avoids rasterio geotransform warning)
            # Fall back to rasterio for georeferenced files
            if p.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    render_simple_image(paths[0], args)
                    return
                except Exception:
                    pass
            render_raster(paths, args)
            return

        elif p.endswith(_SUPPORTED_VECTOR_EXTS):
            if args.table:
                # Treat as tabular data
                try:
                    if p.endswith(('.parquet', '.geoparquet')):
                        df = pd.read_parquet(paths[0])
                    else:
                        import geopandas as gpd
                        gdf = gpd.read_file(paths[0])
                        df = pd.DataFrame(gdf.drop(columns='geometry'))
                    handle_tabular_data(df, args, paths[0])
                    return
                except Exception as e:
                    logger.error(f"[ERROR] Failed to read file: {e}")
                    sys.exit(1)
            else:
                render_vector(paths[0], args)
                return

        if os.path.isdir(paths[0]) and args.gallery:
            render_gallery(paths[0], args, is_vector=False)
            return

        elif p.endswith((".csv", ".parquet")):
            # Try geoparquet first if it's a parquet file
            if p.endswith(".parquet"):
                try:
                    import geopandas as gpd
                    gdf = gpd.read_parquet(paths[0])
                    if hasattr(gdf, 'geometry') and gdf.geometry is not None:
                        render_vector(paths[0], args)
                        return
                except Exception:
                    pass
            
            # Read as tabular data
            try:
                if p.endswith(".parquet"):
                    df = pd.read_parquet(paths[0])
                else:
                    df = pd.read_csv(paths[0])
            except ImportError:
                logger.error("[ERROR] Parquet support requires pyarrow. Install with: pip install pyarrow")
                sys.exit(1)
            except Exception as e:
                logger.error(f"[ERROR] Failed to read file: {e}")
                sys.exit(1)
            
            handle_tabular_data(df, args, paths[0])
            return

        else:
            logger.error("[ERROR] Unsupported file type.")
            sys.exit(1)

    elif len(paths) == 3 and all(p.lower().endswith(_SUPPORTED_RASTER_EXTS) for p in paths):
        render_raster(paths, args)

    else:
        logger.error("[ERROR] Provide one raster/vector file or three rasters for RGB.")
        sys.exit(1)


if __name__ == "__main__":
    main()
