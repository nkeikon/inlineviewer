#!/usr/bin/env python3
"""
viewinline — quick-look geospatial viewer for iTerm2 / ANSI/ASCII preview.

Supports:
  • Rasters (.tif, .tiff, .png, .jpg, .jpeg)
  • Vectors (.shp, .geojson, .gpkg)
  • ANSI color preview in text-only terminals (half-block resolution)

Notes:
  - iTerm2 inline images require ITERM_SESSION_ID.
  - In HPC/text-only shells, switches to ANSI color preview.
"""

import sys, os, base64, shutil, argparse
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from matplotlib import colormaps
import matplotlib as mpl

import warnings

warnings.filterwarnings("ignore", message="More than one layer found", category=UserWarning)

__version__ = "0.1.5"

AVAILABLE_COLORMAPS = [
    "viridis", "inferno", "magma", "plasma",
    "cividis", "terrain", "RdYlGn", "coolwarm",
    "Spectral", "cubehelix", "tab10", "turbo"
]

# ---------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------
def show_inline_image(image_array: np.ndarray, display_scale: float | None = None, is_vector: bool = False) -> None:
    """Display a numpy RGB image inline in iTerm2, with different scaling logic for raster vs vector."""
    try:
        # buffer = BytesIO()
        # Image.fromarray(image_array).save(buffer, format="PNG")
        # encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        buffer = BytesIO()
        # Use lower compression level for faster encoding (~4× faster)
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
    except Exception as e:
        print(f"[WARN] Inline display failed ({e})")



def show_ansi_preview(image_array: np.ndarray, width: int = 120, height: int = 60) -> None:
    """ANSI preview using half-block characters (▀). Experimental"""
    try:
        img = Image.fromarray(image_array).resize((width, height * 2), Image.BILINEAR)
        arr = np.array(img)
        for y in range(0, arr.shape[0] - 1, 2):
            top, bottom = arr[y], arr[y + 1]
            line = "".join(
                f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀"
                for (r1, g1, b1), (r2, g2, b2) in zip(top, bottom)
            )
            print(f"{line}\033[0m")

        # sys.stdout.flush()
        print("[OK] ANSI preview displayed.")
    except Exception as e:
        print(f"[WARN] ANSI preview failed ({e}); saving file...")
        save_image_to_tmp(image_array)


def save_image_to_tmp(image_array: np.ndarray) -> str:
    """Save to /tmp and print file path."""
    outfile = "/tmp/viewinline_preview.png"
    Image.fromarray(image_array).save(outfile)
    print(f"[WARN] Inline not supported — saved preview to {outfile}")
    return outfile


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
def preview_csv(path: str, max_rows: int = 10) -> None:
    """Preview a CSV file"""
    import csv, os
    from itertools import islice

    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            # Peek ahead to get sample rows and total count
            rows = list(islice(reader, max_rows))
            total_rows = sum(1 for _ in reader) + len(rows) + 1  # approximate count

        if not header:
            print(f"[ERROR] No header found in '{os.path.basename(path)}'")
            return

        n_cols = len(header)
        print(f"[DATA] CSV file: {os.path.basename(path)} — {total_rows:,} rows × {n_cols} columns")

        # Show compact summary of columns
        # preview_names = ", ".join(header[:8]) + (" ..." if n_cols > 8 else "")
        preview_names = ", ".join(header)
        # print(f"[INFO] Columns: {preview_names}")

        # --- New section here ---
        import re
        numeric_cols, categorical_cols = [], []
        for col in header:
            col_values = [r[i] for r in rows for i, h in enumerate(header) if h == col and r[i].strip() != ""]
            sample = col_values[:20]
            numeric_like = 0
            for v in sample:
                if re.match(r"^-?\d+(\.\d+)?$", v):
                    numeric_like += 1
            if sample and numeric_like / len(sample) >= 0.8:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # --- Pretty two-column table ---
        max_len = max(len(numeric_cols), len(categorical_cols))
        num_list = numeric_cols + [""] * (max_len - len(numeric_cols))
        cat_list = categorical_cols + [""] * (max_len - len(categorical_cols))

        num_header = "Numeric columns"
        cat_header = "Categorical columns"

        num_width = max(len(num_header), max((len(c) for c in num_list), default=0))
        cat_width = max(len(cat_header), max((len(c) for c in cat_list), default=0))

        sep = f"+{'-'*(num_width+2)}+{'-'*(cat_width+2)}+"
        print(sep)
        print(f"| {num_header:<{num_width}} | {cat_header:<{cat_width}} |")
        print(sep)
        for n, c in zip(num_list, cat_list):
            print(f"| {n:<{num_width}} | {c:<{cat_width}} |")
        print(sep)


        # Ask user if they want to show the first rows
        ans = input("Preview first 10 rows? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            return

        # Compute column widths based on sample
        sample = [header] + rows
        col_widths = [min(max(len(str(cell)) for cell in col), 22) for col in zip(*sample)]

        # Build table
        def fmt_row(row):
            return "| " + " | ".join(f"{str(c)[:w]:<{w}}" for c, w in zip(row, col_widths)) + " |"

        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        print(sep)
        print(fmt_row(header))
        print(sep)
        for r in rows:
            print(fmt_row(r))
        print(sep)

        if len(rows) == max_rows:
            print(f"[INFO] Showing first {max_rows} rows (truncated).")

    except Exception as e:
        print(f"[ERROR] Failed to preview CSV: {e}")


def describe_csv(path: str, column: str = None) -> None:
    """Compute simple descriptive stats for numeric columns (no pandas)."""
    import csv
    import math
    import statistics

    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)

        if not data:
            print("[WARN] No data rows found.")
            return

        # Collect numeric columns
        numeric_cols = {}
        for row in data:
            for key, val in row.items():
                try:
                    num = float(val)
                    if math.isfinite(num):
                        numeric_cols.setdefault(key, []).append(num)
                except (ValueError, TypeError):
                    continue

        if not numeric_cols:
            print("[INFO] No numeric columns found.")
            return

        # --- If a specific column is requested ---
        if column:
            if column not in numeric_cols:
                print(f"[WARN] Column '{column}' not found or not numeric.")
                return
            numeric_cols = {column: numeric_cols[column]}
            print(f"[SUMMARY] Column '{column}' (describe):")
        else:
            print(f"[SUMMARY] Numeric columns (describe):")

        # Prepare table header
        headers = ["Column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        col_widths = [12, 8, 10, 10, 10, 10, 10, 10, 10]
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

        print(sep)
        print(fmt.format(*headers))
        print(sep)

        # Compute and print stats per column
        for name, vals in numeric_cols.items():
            vals.sort()
            n = len(vals)
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if n > 1 else 0
            q25 = vals[int(0.25 * (n - 1))]
            q50 = vals[int(0.5 * (n - 1))]
            q75 = vals[int(0.75 * (n - 1))]
            row = [
                name[:12],
                n,
                f"{mean:.3f}",
                f"{std:.3f}",
                f"{min(vals):.3f}",
                f"{q25:.3f}",
                f"{q50:.3f}",
                f"{q75:.3f}",
                f"{max(vals):.3f}",
            ]
            print(fmt.format(*row))

        print(sep)

    except Exception as e:
        print(f"[ERROR] Failed to describe CSV: {e}")


def inline_histogram_csv(path: str, column: str = None, bins: int = 20, args=None) -> None:
    """Render inline histograms for numeric CSV columns (single or multiple)."""
    import csv, math, io, base64, sys
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import colormaps

    try:
        # --- Read CSV ---
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        if not data:
            print("[WARN] No data found.")
            return

        # --- Extract numeric columns ---
        numeric_cols = {}
        for row in data:
            for key, val in row.items():
                try:
                    num = float(val)
                    if math.isfinite(num):
                        numeric_cols.setdefault(key, []).append(num)
                except (ValueError, TypeError):
                    continue
        if not numeric_cols:
            print("[INFO] No numeric columns found.")
            return

        # --- Select column(s) ---
        if column:
            if column not in numeric_cols:
                print(f"[WARN] Column '{column}' not found or not numeric.")
                return
            cols = [(column, numeric_cols[column])]
        else:
            cols = list(numeric_cols.items())
            print(f"[INFO] Found {len(cols)} numeric columns. Rendering inline histograms...")

        # --- Settings ---
        w, h = 300, 180  # per histogram
        margin = 40
        bg_color = (220, 220, 220)
        text_color = (30, 30, 30)
        font = None
        try:
            font = ImageFont.load_default()
        except Exception:
            pass

        # --- Layout ---
        if len(cols) == 1:
            per_row = 1
            w, h = 400, 200   # make single plot a bit larger
        else:
            per_row = 2
            w, h = 300, 180

        nrows = math.ceil(len(cols) / per_row)
        total_w = per_row * w + (per_row + 1) * margin
        total_h = nrows * h + (nrows + 1) * margin
        canvas = Image.new("RGB", (total_w, total_h), bg_color)
        draw = ImageDraw.Draw(canvas)

        # --- Draw each histogram ---
        for i, (col, vals) in enumerate(cols):
            row_i = i // per_row
            col_i = i % per_row
            x0 = margin + col_i * (w + margin)
            y0 = margin + row_i * (h + margin)
            counts, _ = np.histogram(vals, bins=bins)
            counts = counts.astype(float)
            counts /= counts.max() if counts.max() else 1

            cmap = colormaps["viridis"]
            bw = w / bins
            base_y = y0 + h - 25
            max_count = max(counts) if counts.max() else 1

            for j, c in enumerate(counts):
                bar_h = int(c * (h - 50))
                x1 = int(x0 + j * bw)
                x2 = int(x1 + bw - 1)
                y1 = int(base_y - bar_h)
                normalized_height = c / max_count
                color = tuple(int(x * 255) for x in cmap(normalized_height)[:3])
                draw.rectangle([x1, y1, x2, base_y], fill=color)

            # Axes & labels
            draw.line([x0, base_y, x0 + w, base_y], fill=(120, 120, 120), width=1)
            draw.line([x0, y0 + 25, x0, base_y], fill=(120, 120, 120), width=1)
            draw.text((x0 + 4, y0 + 4), col[:16], fill=text_color, font=font)
            mn, mx = min(vals), max(vals)
            draw.text((x0 + 2, base_y + 5), f"{mn:.1f}", fill=(90, 90, 90), font=font)
            draw.text((x0 + w - 35, base_y + 5), f"{mx:.1f}", fill=(90, 90, 90), font=font)

        # --- Inline display (valid PNG) ---
        if "ITERM_SESSION_ID" in os.environ:
            buf = io.BytesIO()
            canvas.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode()
            w, h = canvas.size
            sys.stdout.write(f"\033]1337;File=inline=1;width={w}px;height={h}px;size={len(data)}:{data}\a\n")
            sys.stdout.flush()
        else:
            canvas.show()

    except Exception as e:
        print(f"[ERROR] Failed to render inline histograms: {e}")


def plot_scatter_csv(path: str, x_col: str, y_col: str, args=None) -> None:
    """A minimal, Pillow-based scatter plot"""
    import pandas as pd
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import io

    try:
        df = pd.read_csv(path)

        if x_col not in df.columns or y_col not in df.columns:
            print(f"[ERROR] Columns '{x_col}' or '{y_col}' not found.")
            print(f"[INFO] Available columns: {', '.join(df.columns)}")
            return

        # --- Filter and sanitize numeric values ---
        df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            print("[WARN] No numeric values to plot.")
            return

        # --- Settings (tight and consistent with histograms) ---
        w, h = 420, 300  # smaller, compact
        margin = 40
        bg_color = (220, 220, 220)
        axis_color = (100, 100, 100)
        point_color = (70, 130, 180)
        text_color = (25, 25, 25)

        img = Image.new("RGB", (w, h), bg_color)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # --- Normalize data ---
        x_vals, y_vals = df[x_col].to_numpy(), df[y_col].to_numpy()
        x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)

        def scale_x(x):
            return margin + (x - x_min) / (x_max - x_min) * (w - 2 * margin)

        def scale_y(y):
            return h - margin - (y - y_min) / (y_max - y_min) * (h - 2 * margin)

        # --- Axes only (no interior grid) ---
        draw.line([(margin, h - margin), (w - margin, h - margin)], fill=axis_color, width=1)
        draw.line([(margin, h - margin), (margin, margin)], fill=axis_color, width=1)

        # --- Scatter points ---
        for x, y in zip(x_vals, y_vals):
            px, py = scale_x(x), scale_y(y)
            s = 1.5  # point size
            # draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=point_color) #regular dots
            draw.line([(px - s, py), (px + s, py)], fill=point_color, width=1)
            draw.line([(px, py - s), (px, py + s)], fill=point_color, width=1)

        # --- Axis labels ---
        draw.text((margin + 2, h - margin + 8), x_col[:14], fill=text_color, font=font)
        draw.text((6, margin - 15), y_col[:14], fill=text_color, font=font)

        # --- Inline display ---
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        from numpy import array
        show_image_auto(np.array(Image.open(buf)), args if args else argparse.Namespace(display=None, ansi_size=None))

    except Exception as e:
        print(f"[ERROR] Scatter plot failed: {e}")

# ---------------------------------------------------------------------
# Raster handling
# ---------------------------------------------------------------------
def normalize_to_uint8(band: np.ndarray) -> np.ndarray:
    band = band.astype(float)
    valid = np.isfinite(band)

    if not np.any(valid):
        return np.zeros_like(band, dtype=np.uint8)

    valid_vals = band[valid]  # ← store once

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


def render_raster(paths: list[str], args) -> None:
    try:
        import rasterio
        import rasterio.enums
    except ImportError:
        print("[ERROR] rasterio not installed. Please install with `pip install rasterio`.")
        return

    try:
        if len(paths) == 1:
            # --- Quick display for standard images (PNG, JPG, JPEG, TIFF)
            ext = os.path.splitext(paths[0].lower())[1]
            if ext in [".png", ".jpg", ".jpeg"]:
                try:
                    # from PIL import Image
                    # img = Image.open(paths[0]).convert("RGB")
                    
                    img = Image.open(paths[0])
                    img.draft("RGB", (1200, 1200))  # decode at lower res if possible
                    img = img.convert("RGB")

                    print(f"[DATA] Image loaded: {os.path.basename(paths[0])} ({img.width}×{img.height})")

                    if args.display:
                        new_w = int(img.width * args.display)
                        new_h = int(img.height * args.display)
                        img = img.resize((new_w, new_h), Image.BILINEAR)
                        print(f"[VIEW] Manual resize ×{args.display:.2f} → {new_w}×{new_h}px")
                    else:
                        img, scale = resize_to_terminal(np.array(img))
                        print(f"[VIEW] Auto-fit display → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")

                    show_image_auto(np.array(img), args)
                    return  # stop here — no rasterio needed
                except Exception as e:
                    print(f"[ERROR] Failed to render image file: {e}")
                    return

            with rasterio.open(paths[0]) as ds:
                H, W = ds.height, ds.width
                print(f"[DATA] Raster loaded: {os.path.basename(paths[0])} ({W}×{H})")

                max_dim = 2000
                if max(H, W) > max_dim:
                    scale = max_dim / max(H, W)
                    out_h, out_w = int(H * scale), int(W * scale)
                    data = ds.read(
                        out_shape=(ds.count, out_h, out_w),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    print(f"[PROC] Downsampled → {out_w}×{out_h}px (scale={scale:.3f})")
                else:
                    data = ds.read()

            # Handle multi-band rasters
            band_count = data.shape[0]
            if band_count >= 3:
                print(f"[INFO] Multi-band raster detected ({band_count} bands).")

                # List available bands
                band_list = ", ".join([str(i + 1) for i in range(band_count)])
                print(f"[INFO] Available bands: {band_list}")

                # Parse --rgb-bands if provided
                if getattr(args, "rgb_bands", None):
                    try:
                        rgb_idx = [int(b) - 1 for b in args.rgb_bands.split(",")]
                        if len(rgb_idx) != 3 or any(i < 0 or i >= band_count for i in rgb_idx):
                            raise ValueError
                        print(f"[INFO] Using specified RGB bands: {args.rgb_bands}")
                    except Exception:
                        print("[WARN] Invalid --rgb-bands format. Expected like '3,2,1'. Using default 1–3.")
                        rgb_idx = [0, 1, 2]
                else:
                    rgb_idx = [0, 1, 2]
                    print("[INFO] Defaulting to first three bands (1–3) for RGB display.")

                rgb = np.stack([normalize_to_uint8(data[i]) for i in rgb_idx], axis=-1)
                img = rgb

            else:
                # Single-band grayscale or colormap
                band_idx = max(0, min(args.band - 1, band_count - 1))
                band = normalize_to_uint8(data[band_idx])

                print(f"[INFO] Displaying band {band_idx + 1} of {band_count}")

                if args.colormap:
                    cmap_name = args.colormap or "terrain"
                    cmap = colormaps[cmap_name]
                    colored = cmap(band / 255.0)
                    img = (colored[:, :, :3] * 255).astype(np.uint8)
                    print(f"[INFO] Applying colormap: {cmap_name}")
                else:
                    img = np.stack([band] * 3, axis=-1)
                    print("[INFO] Displaying in grayscale (no colormap applied)")

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
            print(f"[VIEW] Auto-fit display → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")

        # show_image_auto(img, args)
        show_inline_image(img, getattr(args, "display", None), is_vector=False)


    except Exception as e:
        print(f"[ERROR] Raster rendering failed: {e}")

def render_gallery(folder: str, grid: str = "4x4", args=None) -> None:
    """Render a folder of rasters/images as small thumbnails in a grid."""
    import math
    from PIL import Image
    import numpy as np
    import os

    try:
        # --- Parse grid ---
        try:
            cols, rows = map(int, grid.lower().split("x"))
        except Exception:
            cols, rows = 4, 4
        nmax = cols * rows

        # --- Collect files ---
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(exts)]
        if not files:
            print(f"[WARN] No image/raster files found in {folder}")
            return

        # --- Limit total images to grid size ---
        files = files[:nmax]
        # print(f"[INFO] Showing {len(files)} images ({cols}×{rows} grid)")

        # --- Load each file as RGB thumbnail ---
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

        # --- Create adaptive grid canvas (gray background) ---
        n = len(thumbs)
        cols = min(cols, n)
        rows = math.ceil(n / cols)
        w, h = thumb_size
        margin = 8  # slightly tighter spacing
        canvas_w = cols * w + (cols + 1) * margin
        canvas_h = rows * h + (rows + 1) * margin
        canvas = Image.new("RGB", (canvas_w, canvas_h), (220, 220, 220))  # gray background

        for i, img in enumerate(thumbs):
            r, c = divmod(i, cols)
            x = margin + c * (w + margin)
            y = margin + r * (h + margin)
            canvas.paste(img, (x, y))

        print(f"[INFO] Displaying {n} images ({cols}×{rows} grid)")
        show_image_auto(np.array(canvas), args if args else argparse.Namespace(display=None, ansi_size=None))

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
        # layers = list_layers(path)
        if path.lower().endswith((".shp", ".geojson", ".json", ".parquet", ".geoparquet")):
            # Common single-layer formats — skip list_layers() call
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

    try:
        gdf = gpd.read_file(path, layer=getattr(args, "layer", None))
        print(f"[DATA] Vector loaded: {os.path.basename(path)} ({len(gdf)} features)")
    except Exception as e:
        print(f"[ERROR] Failed to read vector: {e}")
        return

    # Detect numeric columns
    num_cols = []
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        try:
            if np.issubdtype(gdf[c].dtype, np.number):
                num_cols.append(c)
        except TypeError:
            continue

    if num_cols:
        print("[INFO] Numeric columns detected:", ", ".join(num_cols))
        if not args.color_by:
            print("[INFO] Showing border-only view (use --color-by <column> to color by numeric values).")
    else:
        print("[INFO] Displaying boundaries only - no numeric columns detected")

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor="gray")
    ax.set_facecolor("gray")
    ax.set_axis_off()

    # Determine colormap
    column = args.color_by if args.color_by in gdf.columns else None

    # Warn if user provided an invalid column
    if args.color_by and args.color_by not in gdf.columns:
        print(f"[WARN] Column '{args.color_by}' not found. Showing border-only view.")
        column = None

    if column and args.colormap is None:
        args.colormap = "terrain"
        print("[INFO] Applying default colormap: terrain")

    cmap = colormaps.get(args.colormap) if args.colormap else None

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    try:
        if column and np.issubdtype(gdf[column].dtype, np.number):
            vmin, vmax = np.percentile(gdf[column].dropna(), (2, 98))
            print(f"[INFO] Coloring by '{column}' (range: {vmin:.2f}–{vmax:.2f})")

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = colormaps.get(args.colormap or "terrain")

            geom_type = gdf.geom_type.iloc[0]

            if geom_type.startswith("Line"):
                # Lines — colored directly by value, no fill
                for _, row in gdf.iterrows():
                    val = row[column]
                    color = cmap(norm(val))
                    ax.plot(*row.geometry.xy, color=color, linewidth=getattr(args, "width", 0.7))
            else:
                # Polygons — color fill with thin white outlines (like viewgeom dark theme)
                gdf.plot(
                    ax=ax,
                    column=column,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    linewidth=0.2,
                    edgecolor="white",   # match viewgeom’s dark-theme outline
                    zorder=1
                )

        else:
            # Default: outline-only mode (no fill)
            gdf.plot(
                ax=ax,
                facecolor="none",
                edgecolor=args.edgecolor,  # default #F6FF00
                linewidth=getattr(args, "width", 0.7),
                zorder=1
            )

    except Exception as e:
        print(f"[WARN] Plotting failed ({e}) — fallback to border-only.")
        gdf.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.5)


    # Save to buffer (adaptive DPI)
    render_dpi = 200 if "ITERM_SESSION_ID" in os.environ else 400
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=render_dpi,
                bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    # print(f"[PROC] Rendered vector") # (DPI={render_dpi})

    # --- Resize for terminal or manual override ---
    if getattr(args, "display", None):
        scale = float(args.display)
        new_w = max(1, int(img.shape[1] * scale))
        new_h = max(1, int(img.shape[0] * scale))
        img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
        print(f"[VIEW] Manual resize ×{scale:.2f} → {new_w}×{new_h}px")
    else:
        # --- Resize for terminal or manual display ---
        if args.display:
            new_w, new_h = max(1, int(img.shape[1] * args.display)), max(1, int(img.shape[0] * args.display))
            img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
            print(f"[VIEW] Manual resize ×{args.display:.2f} → {new_w}×{new_h}px")
        else:
            img, scale = resize_to_terminal(img)
            print(f"[VIEW] Auto-fit display → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")

    show_inline_image(img, getattr(args, "display", None), is_vector=True)

# ---------------------------------------------------------------------
# Smart display selector
# ---------------------------------------------------------------------
def show_image_auto(img: np.ndarray, args) -> None:
    """Automatically pick best display method."""
    if "ITERM_SESSION_ID" in os.environ:
        try:
            show_inline_image(img, getattr(args, "display", None))
            # print("[OK] Inline render complete.")
            return
        except Exception:
            print("[WARN] Inline display failed; trying ANSI fallback...")

    if sys.stdout.isatty(): #and os.getenv("TERM"):
        try:
            w, h = (120, 60)
            mode = "auto"
            if getattr(args, "ansi_size", None):
                try:
                    w, h = map(int, args.ansi_size.lower().split("x"))
                    mode = f"ansi-size {w}x{h}"
                except Exception:
                    pass
            elif getattr(args, "display", None):
                w = max(1, int(w * args.display))
                h = max(1, int(h * args.display))
                mode = f"display size {args.display:.2f}"
            print(f"[VIEW] ANSI display → {w}×{h} grid ({mode})")
            show_ansi_preview(img, width=w, height=h)
            return
        except Exception as e:
            print(f"[WARN] ANSI preview failed ({e}); saving file...")

    save_image_to_tmp(img)

# ---------------------------------------------------------------------
# Smart help formatter to hide None defaults
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
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="viewinline",
        description=(
            "Quick-look geospatial viewer for iTerm2 and headless environments.\n\n"
            "Supports rasters (.tif, .tiff, .png, .jpg, .jpeg), "
            "vectors (.shp, .geojson, .gpkg), and CSV preview.\n"
            "Displays inline in iTerm2 if available, otherwise as ANSI color preview."
        ),
        formatter_class=SmartDefaults
    )

    # File input
    parser.add_argument(
        "paths", nargs="+",
        help="Path to raster(s), vector, or CSV file. Provide 1 file or exactly 3 rasters for RGB (R G B)."
    )

    # Display options
    parser.add_argument(
        "--display", type=float, default=None,
        help="Resize only the displayed image (0.5=smaller, 2=bigger). Default: auto-fit to terminal."
    )
    parser.add_argument(
        "--ansi-size", type=str, default=None,
        help="ANSI fallback resolution. Try 180x90 or 200x100."
    )

    # Raster options
    parser.add_argument(
        "--band", type=int, default=1,
        help="Band number to display (single raster case)."
    )
    parser.add_argument(
        "--colormap", nargs="?", const="terrain",
        choices=AVAILABLE_COLORMAPS, default=None,
        help="Apply colormap to single-band rasters or vector coloring. Flag without value → 'terrain'."
    )
    parser.add_argument(
        "--rgb-bands", type=str, default=None,
        help="Comma-separated band numbers for RGB display (e.g., '3,2,1'). Overrides default 1-3."
    )
    parser.add_argument(
    "--gallery", nargs="?", const="4x4", metavar="GRID",
    help="Display all PNG/JPG/TIF images in a folder as thumbnails (e.g., 5x5 grid)."
)

    # CSV options
    parser.add_argument(
    "--hist",
    nargs="?",            # <- makes the argument optional
    const=True,           # <- allows `--hist` with no value
    help="Show histograms for all numeric columns or specify one column name."
)
    parser.add_argument(
        "--describe",
        nargs="?",            # <- same idea
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
        help="Layer name for GeoPackage or multi-layer files."
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # --- Basic argument sanity check ---
    for bad in ("color-by", "edgecolor", "colormap", "band", "display"):
        for a in args.paths:
            if a == bad:
                print(f"[ERROR] Missing '--' before '{bad}'.")
                print("        Example:  --color-by column_name")
                sys.exit(1)

    paths = args.paths
    term_program = os.environ.get("TERM_PROGRAM", "").lower()

    if "iterm" not in term_program:
        print("[WARN] iTerm2 not detected. For better inline image display, use iTerm2 (mac).")
        print("[INFO] Switching to ANSI/ASCII preview mode. This may not display correctly on all terminals.")
        try:
            ans = input("Continue with ANSI/ASCII preview? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("Cancelled by user.")
            sys.exit(0)

    # -----------------------------------------------------------------
    # File routing
    # -----------------------------------------------------------------
    raster_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    vector_exts = (".shp", ".geojson", ".json", ".gpkg")

    if len(paths) == 1:
        p = paths[0].lower()

        if p.endswith(raster_exts):
            render_raster(paths, args)
            return

        elif p.endswith(vector_exts):
            render_vector(paths[0], args)
            return

        if os.path.isdir(paths[0]) and args.gallery:
            render_gallery(paths[0], grid=args.gallery, args=args)
            return

        elif p.endswith(".csv"):

            # --- Handle CSV unique values ---
            if args.unique:
                import csv

                col = args.unique
                with open(paths[0], newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if col not in reader.fieldnames:
                        print(f"[ERROR] Column '{col}' not found in dataset.")
                        sys.exit(1)
                    unique_vals = set()
                    for row in reader:
                        val = row.get(col)
                        if val not in (None, "", " "):
                            unique_vals.add(val)
                print(f"[DATA] CSV file: {os.path.basename(paths[0])}")
                print(f"[DATA] Unique values in '{col}' ({len(unique_vals)}):")

                vals = sorted(unique_vals)
                n = len(vals)

                # Auto-select number of columns
                ncols = 1 if n <= 10 else (2 if n <= 30 else 3)
                nrows = (n + ncols - 1) // ncols

                # Pad values to fill grid evenly
                vals += [""] * (nrows * ncols - n)

                # Print in columns
                col_width = max(len(str(v)) for v in vals if v) + 2
                for i in range(nrows):
                    row = "".join(str(vals[i + j * nrows]).ljust(col_width) for j in range(ncols))
                    print("  " + row.strip())

                sys.exit(0)  # this ensures no preview prompt follows

            if args.scatter:
                plot_scatter_csv(paths[0], args.scatter[0], args.scatter[1], args=args)
                return

            if args.describe or args.hist:
                # --- Summary statistics ---
                if args.describe:
                    if isinstance(args.describe, str):
                        # describe specific column
                        col = args.describe
                        describe_csv(paths[0], column=col)
                    else:
                        # describe all numeric columns
                        describe_csv(paths[0])

                # --- Histograms ---
                if args.hist:
                    if isinstance(args.hist, str):
                        # single column histogram
                        inline_histogram_csv(paths[0], column=args.hist, bins=args.bins, args=args)
                    else:
                        # all numeric columns
                        inline_histogram_csv(paths[0], bins=args.bins, args=args)

            else:
                # Default behavior: show preview only
                preview_csv(paths[0])
                print("[INFO] Use --describe for summary or --hist for histograms.")

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
