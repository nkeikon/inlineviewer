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

import warnings

warnings.filterwarnings("ignore", message="More than one layer found", category=UserWarning)
warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

__version__ = "0.2.1"

AVAILABLE_COLORMAPS = [
    "viridis", "inferno", "magma", "plasma",
    "cividis", "terrain", "RdYlGn", "coolwarm",
    "Spectral", "cubehelix", "tab10", "turbo"
]

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
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

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

    sys.stdout.write(f"\033]1337;File=inline=1;width={width_pct}%:{encoded}\a\n")
    sys.stdout.flush()


def show_image_auto(img: np.ndarray, display_scale = None, is_vector: bool = False) -> None:
    """Attempt inline image display. No fallbacks, no detection.
    
    Just sends the iTerm2 inline image escape sequence. If the terminal supports it,
    great. If not, the escape codes are ignored and nothing happens.
    
    """
    if os.environ.get("TMUX"):
        print("[WARN] Inside tmux — inline images won't display even with allow-passthrough on (known iTerm2/tmux issue). Use a plain terminal tab.")
        return
    
    try:
        show_inline_image(img, display_scale, is_vector)
        print("[VIEW] Image sent — visible in compatible terminals")
    except Exception as e:
        # If image encoding fails, print error but don't crash
        print(f"[ERROR] Failed to encode image: {e}")
        import traceback
        traceback.print_exc()


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
        print(f"[ERROR] Failed to read CSV: {e}")
        return pd.DataFrame()


# =============================================================
# Preview
# =============================================================
def preview_df(df, max_rows: int = 10, query_mode: bool = False, filename: str = None) -> None:
    """Preview a pandas DataFrame."""

    if df is None or df.empty:
        print("[WARN] No rows to preview.")
        return

    n_rows, n_cols = df.shape

    # -------------------------------------------------------------
    # Print dataset header (ONLY when not query mode)
    # -------------------------------------------------------------
    if not query_mode:
        name = filename if filename else "DataFrame"
        print(f"[DATA] CSV file: {name} — {n_rows:,} rows × {n_cols} columns")

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

    print(sep)
    print(fmt_row(columns))
    print(sep)

    for _, r in rows_to_show.iterrows():
        print(fmt_row(r.tolist()))

    print(sep)

    print("[INFO] Use --describe for summary or --hist for histograms.")

# =============================================================
# Describe
# =============================================================
def describe_df(df, column=None):
    if df is None or df.empty:
        print("[WARN] No data rows found.")
        return

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("[INFO] No numeric columns found.")
        return

    if column:
        if column not in numeric_df.columns:
            print(f"[WARN] Column '{column}' not numeric.")
            return
        numeric_df = numeric_df[[column]]
        print(f"[SUMMARY] Column '{column}' (describe):")
    else:
        print("[SUMMARY] Numeric columns (describe):")

    headers = ["Column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    col_widths = [12, 8, 10, 10, 10, 10, 10, 10, 10]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print(sep)
    print(fmt.format(*headers))
    print(sep)

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

        print(fmt.format(*row))

    print(sep)

# =============================================================
# Histogram
# =============================================================
def inline_histogram_df(df, column=None, bins=20, display_scale=None, is_vector=False):

    if df is None or df.empty:
        print("[WARN] No data to plot.")
        return

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("[INFO] No numeric columns found.")
        return

    if column:
        if column not in numeric_df.columns:
            print(f"[WARN] Column '{column}' not numeric.")
            return
        cols = [(column, numeric_df[column].dropna().values)]
    else:
        cols = [(c, numeric_df[c].dropna().values) for c in numeric_df.columns]

    draw_histograms(cols, bins, display_scale, is_vector)


def draw_histograms(cols, bins, display_scale=None, is_vector=False):
    """Render histograms and display via unified show_image_auto()"""
    import math

    if not cols:
        print("[INFO] No numeric columns found.")
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
        print(f"[INFO] Column: {col}")
        print(f"       Most frequent range: {edges[idx]:.2f} – {edges[idx+1]:.2f}")
        print(f"       Values in this range: {counts[idx]}")

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
        print("[WARN] No data to plot.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        print(f"[ERROR] Columns '{x_col}' or '{y_col}' not found.")
        return

    df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        print("[WARN] No numeric values to plot.")
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
        print("[WARN] Scatter plot requires at least two distinct values.")
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
        print(f"[VIEW] Using manual scaling: {mn} to {mx}")

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
        print(f"[DATA] Image loaded: {os.path.basename(filepath)} ({W}×{H})")
        
        if args.display:
            new_w, new_h = max(1, int(W * args.display)), max(1, int(H * args.display))
            img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.BILINEAR))
            print(f"[VIEW] Manual resize ×{args.display:.2f} → {new_w}×{new_h}px")
        else:
            img_array, scale = resize_to_terminal(img_array)
            print(f"[VIEW] Rendered image size → {img_array.shape[1]}×{img_array.shape[0]}px (size={scale:.2f})")
        
        show_image_auto(img_array, getattr(args, "display", None), is_vector=False)
        
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")

def render_raster(paths: list[str], args) -> None:
    try:
        import rasterio
        import rasterio.enums
    except ImportError:
        print("[ERROR] rasterio not installed. Please install with `pip install rasterio`.")
        return

    try:

        if len(paths) == 1:
            path = paths[0]
            
            # Handle NetCDF/HDF with subdatasets
            if path.lower().endswith(('.nc', '.hdf', '.hdf5', '.h5')):
                try:
                    with rasterio.open(path) as src:
                        subdatasets = src.subdatasets
                    
                    # If there are subdatasets, require --subset to select one
                    if subdatasets:
                        if not args.subset:
                            file_type = "variables" if path.lower().endswith('.nc') else "datasets"
                            print(f"Found {len(subdatasets)} {file_type} in {os.path.basename(path)}:")
                            for i, sub in enumerate(subdatasets, 1):
                                # Extract dataset/variable name from GDAL subdataset string
                                ds_name = sub.split(':')[-1].lstrip('/')
                                print(f"  [{i}] {ds_name}")
                            print(f"\nUse --subset <N> to display a specific {file_type[:-1]}.")
                            return
                        
                        # Select by index
                        if args.subset < 1 or args.subset > len(subdatasets):
                            print(f"[ERROR] --subset must be between 1 and {len(subdatasets)}")
                            return
                        
                        path = subdatasets[args.subset - 1]
                        var_name = path.split(':')[-1]
                        print(f"[INFO] Displaying variable {args.subset}: {var_name}")
                
                except rasterio.errors.RasterioIOError as e:
                    # GDAL lacks support, try h5py fallback for HDF5
                    if path.lower().endswith(('.hdf5', '.h5')):
                        try:
                            import h5py
                        except ImportError:
                            print("[ERROR] HDF5 file cannot be opened.")
                            print("[INFO] Requires either:")
                            print("        - GDAL with HDF5 support, or")
                            print("        - h5py: pip install h5py")
                            return
                        
                        # print("[ERROR] h5py fallback not yet implemented.")
                        print("[INFO] Install GDAL with HDF5")
                        return
                    
                    elif path.lower().endswith('.hdf'):
                        print(f"[ERROR] Cannot open HDF4 file: {e}")
                        print("[INFO] HDF4 requires GDAL with HDF4 support")
                        return
                    else:
                        # NetCDF error
                        print(f"[ERROR] Cannot open NetCDF file: {e}")
                        return

            # Continue with normal raster opening
            with rasterio.open(path) as ds:
                H, W = ds.height, ds.width
                print(f"[DATA] Raster loaded: {os.path.basename(paths[0])} ({W}×{H})")
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

                    print(f"[PROC] Downsampled for preview → {out_w}×{out_h}px (scale={scale:.3f})")
                else:
                    data = ds.read()

            # Print band/slice count for all multi-band files
            if band_count > 1:
                if paths[0].lower().endswith('.nc'):
                    print(f"[INFO] {band_count} slices detected")
                else:
                    print(f"[INFO] Multi-band raster detected ({band_count} bands)")

            # MULTI BAND RGB (skip for NetCDF - treat as slices/timesteps, not RGB)
            if band_count >= 3 and not paths[0].lower().endswith('.nc'):


                if getattr(args, "rgb", None):
                    try:
                        rgb_idx = [b - 1 for b in args.rgb]
                        if len(rgb_idx) != 3:
                            raise ValueError
                        print(f"[INFO] Using RGB bands: {args.rgb}")
                    except Exception:
                        print("[WARN] Invalid --rgb. Using default 1 2 3")
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

                band_idx = max(0, min(args.band - 1, band_count - 1))
                # print(f"[INFO] Displaying band {band_idx + 1} of {band_count}")
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
                        print(f"[DATA] Slice {band_idx + 1} of {band_count} — value range: {min_val:.3f} → {max_val:.3f}")
                        if band_count > 1:
                            print(f"[INFO] Use --band <N> to display a different slice")
                    else:
                        print(f"[DATA] Band {band_idx + 1} of {band_count} — value range: {min_val:.3f} → {max_val:.3f}")
                                        
                else:
                    print("[WARN] No valid pixels found.")

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
                    print(f"[INFO] Applying colormap: {args.colormap}")
                else:
                    img = np.stack([band] * 3, axis=-1)
                    print("[INFO] Displaying grayscale")

        elif len(paths) == 3:
            bands = []
            for p in paths:
                with rasterio.open(p) as ds:
                    bands.append(ds.read(1))
            shapes = {b.shape for b in bands}
            if len(shapes) != 1:
                print("[ERROR] Raster sizes do not match.")
                return
            data = np.stack(bands, axis=0)
            H, W = data.shape[1:]
            print(f"[DATA] RGB raster stack loaded: {W}×{H}")
            img = np.stack([normalize_to_uint8(b) for b in data], axis=-1)
            print("[INFO] Displaying 3-band RGB composite")

        else:
            print("[ERROR] Provide one raster or exactly three rasters for RGB.")
            return

        # Resize for terminal
        H, W = img.shape[:2]
        if args.display:
            new_w, new_h = max(1, int(W * args.display)), max(1, int(H * args.display))
            img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
            print(f"[VIEW] Manual resize ×{args.display:.2f} → {new_w}×{new_h}px")
        else:
            img, scale = resize_to_terminal(img)
            print(f"[VIEW] Rendered image size → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")

        # Use unified display
        show_image_auto(img, getattr(args, "display", None), is_vector=False)

    except Exception as e:
        if paths[0].lower().endswith('.nc'):
            print(f"[ERROR] Cannot display this variable.")
            print("[INFO] viewinline only supports 2D or 3D NetCDF variables")
        else:
            print(f"[ERROR] Raster rendering failed: {e}")


def render_gallery(folder: str, grid: str = "4x4", display_scale=None, is_vector=False) -> None:
    """Render a folder of rasters/images as small thumbnails in a grid."""
    import math

    try:
        # Parse grid
        try:
            cols, rows = map(int, grid.lower().split("x"))
        except Exception:
            cols, rows = 4, 4
        nmax = cols * rows

        # Collect files
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(exts)]
        if not files:
            print(f"[WARN] No image/raster files found in {folder}")
            return

        files = files[:nmax]

        # Load thumbnails
        thumbs = []
        thumb_size = (128, 128)
        for f in files:
            try:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".tif", ".tiff"]:
                    import rasterio
                    with rasterio.open(f) as ds:
                        arr = ds.read()
                        if arr.shape[0] >= 3:
                            rgb = np.stack([normalize_to_uint8(arr[i]) for i in range(3)], axis=-1)
                        else:
                            band = normalize_to_uint8(arr[0])
                            rgb = np.stack([band]*3, axis=-1)
                        img = Image.fromarray(rgb)
                else:
                    img = Image.open(f).convert("RGB")
                img.thumbnail(thumb_size)
                thumbs.append(img)
            except Exception as e:
                print(f"[WARN] Skipped {os.path.basename(f)} ({e})")

        if not thumbs:
            print("[WARN] No valid images loaded.")
            return

        # Create grid canvas
        n = len(thumbs)
        cols = min(cols, n)
        rows = math.ceil(n / cols)
        w, h = thumb_size
        margin = 8
        canvas_w = cols * w + (cols + 1) * margin
        canvas_h = rows * h + (rows + 1) * margin
        canvas = Image.new("RGB", (canvas_w, canvas_h), (220, 220, 220))

        for i, img in enumerate(thumbs):
            r, c = divmod(i, cols)
            x = margin + c * (w + margin)
            y = margin + r * (h + margin)
            canvas.paste(img, (x, y))

        print(f"[INFO] Displaying {n} images ({cols}×{rows} grid)")
        show_image_auto(np.array(canvas), display_scale, is_vector)

    except Exception as e:
        print(f"[ERROR] Failed to render gallery: {e}")

# ---------------------------------------------------------------------
# Vector handling
# ---------------------------------------------------------------------
def render_vector(path, args):
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from pyogrio import list_layers
    except ImportError as e:
        print("[ERROR] Missing dependency. Install with:")
        print("  pip install geopandas matplotlib pyogrio")
        return

    try:
        # Handle layer selection
        if path.lower().endswith((".shp", ".geojson", ".json", ".parquet", ".geoparquet")):
            layers = [(os.path.splitext(os.path.basename(path))[0], None)]
        else:
            layers = list_layers(path)
        if len(layers) > 1 and not getattr(args, "layer", None):
            print(f"[INFO] Multiple layers found in '{os.path.basename(path)}':")
            for i, lyr in enumerate(layers, 1):
                name = lyr[0]
                geom = lyr[1] if len(lyr) > 1 and lyr[1] else "Unknown"
                print(f"   {i}. {name} ({geom})")
            first = layers[0][0]
            print(f"[INFO] Defaulting to first layer: '{first}' (use --layer <name> to select another).")
            args.layer = first
    except Exception as e:
        print(f"[WARN] Could not list layers: {e}")

    # try:
    #     gdf = gpd.read_file(path, layer=getattr(args, "layer", None))
    #     print(f"[DATA] Vector loaded: {os.path.basename(path)} ({len(gdf)} features)")
    # except Exception as e:
    #     print(f"[ERROR] Failed to read vector: {e}")
    #     return

    try:
        # Use read_parquet for parquet/geoparquet files
        if path.lower().endswith(('.parquet', '.geoparquet')):
            gdf = gpd.read_parquet(path)
        else:
            gdf = gpd.read_file(path, layer=getattr(args, "layer", None))
        print(f"[DATA] Vector loaded: {os.path.basename(path)} ({len(gdf)} features)")
    except ImportError:
        print("[ERROR] Parquet/GeoParquet support requires pyarrow. Install with: pip install pyarrow")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read vector: {e}")
        return
        
    # Detect non-geometry columns
    all_cols = [c for c in gdf.columns if c != gdf.geometry.name]

    if all_cols:
        n = len(all_cols)
        if not args.color_by:  # Only show columns if user didn't specify one
            print(f"[INFO] Available columns ({n}):")
            if n <= 20:
                for c in all_cols:
                    print(f"  {c}")
            else:
                ncols = 2 if n <= 30 else 3 if n <= 100 else 4
                nrows = (n + ncols - 1) // ncols
                padded = all_cols + [""] * (nrows * ncols - n)
                col_width = max(len(c) for c in all_cols) + 3
                for i in range(nrows):
                    row = ""
                    for j in range(ncols):
                        row += padded[i + j * nrows].ljust(col_width)
                    print("  " + row.rstrip())
            print("[INFO] Showing border-only view (use --color-by <column> to color features).")
    else:
        print("[INFO] No attribute columns found.")
        
    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor="gray")
    ax.set_facecolor("gray")
    ax.set_axis_off()

    # Determine colormap
    column = args.color_by if args.color_by in gdf.columns else None

    if args.color_by and args.color_by not in gdf.columns:
        print(f"[WARN] Column '{args.color_by}' not found. Showing border-only view.")
        column = None

    if column and args.colormap is None:
        args.colormap = "terrain"
        print("[INFO] Applying default colormap: terrain")

    cmap = colormaps.get(args.colormap) if args.colormap else None

    # Plot
    try:
        if column:
            # NUMERIC COLUMN
            if np.issubdtype(gdf[column].dtype, np.number):
                vmin, vmax = np.percentile(gdf[column].dropna(), (2, 98))
                print(f"[INFO] Coloring by numeric column '{column}' (range: {vmin:.2f}–{vmax:.2f})")

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
                print(f"[INFO] Coloring by categorical column '{column}' ({len(categories)} classes)")

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
        print(f"[WARN] Plotting failed ({e}) — fallback to border-only.")
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
        print(f"[VIEW] Manual resize ×{scale:.2f} → {new_w}×{new_h}px")
    else:
        img, scale = resize_to_terminal(img)
        print(f"[VIEW] Rendered image size → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")

    # Use unified display
    show_image_auto(img, getattr(args, "display", None), is_vector=True)

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
        print("[ERROR] --sql cannot be combined with --where/--sort/--limit/--select.")
        sys.exit(1)

    if args.sql:
        try:
            import duckdb
        except ImportError:
            print("[ERROR] --sql requires DuckDB. Install with: pip install duckdb")
            sys.exit(1)

        print("[INFO] Executing SQL query...")

        try:
            query = args.sql.replace("data", f"read_csv_auto('{filepath}')")
            con = duckdb.connect()
            df = con.execute(query).df()
            con.close()
        except Exception as e:
            print(f"[ERROR] DuckDB SQL failed: {e}")
            sys.exit(1)

        if df.empty:
            print("[WARN] Query returned no rows.")
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
            print("[ERROR] Filtering requires DuckDB. Install with: pip install duckdb")
            sys.exit(1)

        print("[INFO] Building query...")

        base_query = "SELECT * FROM df"

        if args.select:
            selected = ", ".join(args.select)
            print(f"[INFO] Selecting columns: {selected}")
            base_query = f"SELECT {selected} FROM df"

        clauses = []

        if args.where:
            print(f"[INFO] Applying filter: {args.where}")
            clauses.append(f"WHERE {args.where}")

        if args.sort:
            direction = "DESC" if args.desc else "ASC"
            print(f"[INFO] Sorting by: {args.sort} ({direction})")
            clauses.append(f"ORDER BY {args.sort} {direction}")

        if args.limit:
            print(f"[INFO] Limiting rows: {args.limit}")
            clauses.append(f"LIMIT {args.limit}")

        query = " ".join([base_query] + clauses)

        try:
            df = duckdb.query(query).to_df()
        except Exception as e:
            print(f"[ERROR] DuckDB query failed: {e}")
            sys.exit(1)

        if df.empty:
            print("[WARN] Query returned no rows.")
            return

    if args.unique:
        col = args.unique

        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not found.")
            return

        vals = sorted(df[col].dropna().astype(str).unique())
        n = len(vals)

        print(f"[DATA] Unique values in '{col}' ({n}):")

        if n == 0:
            print("  (none)")
            return

        if n <= 10:
            for v in vals:
                print(f"  {v}")
        else:
            ncols = 2 if n <= 30 else 3 if n <= 100 else 4
            nrows = (n + ncols - 1) // ncols
            vals += [""] * (nrows * ncols - n)
            col_width = max(len(v) for v in vals) + 3

            for i in range(nrows):
                row = ""
                for j in range(ncols):
                    row += vals[i + j * nrows].ljust(col_width)
                print("  " + row.rstrip())

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
        "--band", type=int, default=1,
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
        "--subset", type=int, default=None,
        help="Variable index for NetCDF files (e.g. --subset 1)."
    )
    parser.add_argument(
        "--rgbfiles", nargs=3, type=str, metavar=('R', 'G', 'B'),
        help="Three single-band rasters for RGB composite (e.g., --rgbfiles R.tif G.tif B.tif). Can also provide as positional arguments without the flag."
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

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

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
                print(f"[ERROR] Missing '--' before '{bad}'.")
                print("        Example:  --color-by column_name")
                sys.exit(1)

    paths = args.paths

    # File routing
    raster_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nc", ".hdf", ".hdf5", ".h5")
    vector_exts = (".shp", ".geojson", ".json", ".gpkg", ".parquet", "geoparquet")

    if len(paths) == 1:
        p = paths[0].lower()

        if p.endswith(raster_exts):
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

        elif p.endswith(vector_exts):
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
                    print(f"[ERROR] Failed to read file: {e}")
                    sys.exit(1)
            else:
                render_vector(paths[0], args)
                return

        if os.path.isdir(paths[0]) and args.gallery:
            render_gallery(paths[0], grid=args.gallery, display_scale=args.display, is_vector=False)
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
                print("[ERROR] Parquet support requires pyarrow. Install with: pip install pyarrow")
                sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Failed to read file: {e}")
                sys.exit(1)
            
            handle_tabular_data(df, args, paths[0])
            return

        else:
            print("[ERROR] Unsupported file type.")
            sys.exit(1)

    elif len(paths) == 3 and all(p.lower().endswith(raster_exts) for p in paths):
        render_raster(paths, args)

    else:
        print("[ERROR] Provide one raster/vector file or three rasters for RGB.")
        sys.exit(1)


if __name__ == "__main__":
    main()
