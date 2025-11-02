#!/usr/bin/env python3
"""
viewinline — quick-look geospatial viewer for iTerm2 / ANSI/ASCII preview.

Supports:
  • Rasters (.tif, .tiff)
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
import warnings

warnings.filterwarnings("ignore", message="More than one layer found", category=UserWarning)

__version__ = "0.1.1"

AVAILABLE_COLORMAPS = [
    "viridis", "inferno", "magma", "plasma",
    "cividis", "terrain", "RdYlGn", "coolwarm",
    "Spectral", "cubehelix", "tab10", "turbo"
]

# ---------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------
def show_inline_image(image_array: np.ndarray, display_scale: float | None = None) -> None:
    """Display a numpy RGB image inline in iTerm2, default width=33%."""
    try:
        buffer = BytesIO()
        Image.fromarray(image_array).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Default: 1/3 of window
        if display_scale is None:
            width_pct = 33
        else:
            # Interpret --display as multiplier of full width (1.0 = 100%)
            width_pct = int(100 * display_scale)
            # Clamp to reasonable range
            width_pct = max(5, min(width_pct, 300))

        sys.stdout.write(f"\033]1337;File=inline=1;width={width_pct}%:{encoded}\a\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"[WARN] Inline display failed ({e})")


def show_ansi_preview(image_array: np.ndarray, width: int = 120, height: int = 60) -> None:
    """ANSI preview using half-block characters (▀)."""
    try:
        img = Image.fromarray(image_array).resize((width, height * 2), Image.BILINEAR)
        arr = np.array(img)
        # for y in range(0, arr.shape[0] - 1, 2):
        #     top = arr[y]
        #     bottom = arr[y + 1]
        #     line = []
        #     for (r1, g1, b1), (r2, g2, b2) in zip(top, bottom):
        #         line.append(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀")
        #     print("".join(line) + "\033[0m")
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
# Raster handling
# ---------------------------------------------------------------------
def normalize_to_uint8(band: np.ndarray) -> np.ndarray:
    band = band.astype(float)
    valid = np.isfinite(band)
    if not np.any(valid):
        return np.zeros_like(band, dtype=np.uint8)
    mn, mx = np.percentile(
        band[valid] if band[valid].size < 1_000_000 else np.random.choice(band[valid], 1_000_000, replace=False),
        (2, 98)
    )
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

            band_idx = max(0, min(args.band - 1, data.shape[0] - 1))
            band = normalize_to_uint8(data[band_idx])
            print(f"[INFO] Displaying band {band_idx + 1} of {data.shape[0]}")

            # Grayscale default
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

        show_image_auto(img, args)

    except Exception as e:
        print(f"[ERROR] Raster rendering failed: {e}")


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

    # Figure setup (black background)
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

    # Plot
    try:
        if column and np.issubdtype(gdf[column].dtype, np.number):
            vmin, vmax = np.percentile(gdf[column].dropna(), (2, 98))
            print(f"[INFO] Coloring by '{column}' (range: {vmin:.2f}–{vmax:.2f})")
            gdf.plot(ax=ax, column=column, cmap=cmap, vmin=vmin, vmax=vmax,
                     linewidth=0.3, edgecolor="black", zorder=1)
        else:
            gdf.plot(ax=ax, facecolor="none", edgecolor=args.edgecolor,
                     linewidth=0.7, zorder=1)
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

    img, scale = resize_to_terminal(img)
    print(f"[VIEW] Auto-fit display → {img.shape[1]}×{img.shape[0]}px (size={scale:.2f})")

    show_image_auto(img, args)


# ---------------------------------------------------------------------
# Smart display selector
# ---------------------------------------------------------------------
def show_image_auto(img: np.ndarray, args) -> None:
    """Automatically pick best display method."""
    if "ITERM_SESSION_ID" in os.environ:
        try:
            show_inline_image(img, getattr(args, "display", None))
            print("[OK] Inline render complete.")
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
            "Supports rasters (.tif, .tiff) and vectors (.shp, .geojson, .gpkg).\n"
            "Displays inline in iTerm2 if available, otherwise as ANSI color preview."
        ),
            formatter_class=SmartDefaults
)

    # File input
    parser.add_argument("paths", nargs="+",
        help="Path to raster(s) or vector file. Provide 1 file or exactly 3 rasters for RGB (R G B).")

    # Display options
    parser.add_argument("--display", type=float, default=None,
        help="Resize only the displayed image (0.5=smaller, 2=bigger). Default: auto-fit to terminal.")
    parser.add_argument("--ansi-size", type=str, default=None,
        help="ANSI fallback resolution. Try 180x90 or 200x100.")

    # Raster options
    parser.add_argument("--band", type=int, default=1,
        help="Band number to display (single raster case).")
    parser.add_argument("--colormap", nargs="?", const="terrain",
        choices=AVAILABLE_COLORMAPS, default=None,
        help="Apply colormap to single-band rasters or vector coloring. Flag without value → 'terrain'.")

    # Vector options
    parser.add_argument("--color-by", type=str, default=None,
        help="Numeric column to color vector features by (optional).")
    parser.add_argument("--edgecolor", type=str, default="#F6FF00",
        help="Edge color for vector outlines (hex or named color).")
    parser.add_argument("--layer", type=str, default=None,
        help="Layer name for GeoPackage or multi-layer files.")

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

    raster_exts = (".tif", ".tiff")
    vector_exts = (".shp", ".geojson", ".json", ".gpkg")

    if len(paths) == 1:
        p = paths[0].lower()
        if p.endswith(raster_exts):
            render_raster(paths, args)
        elif p.endswith(vector_exts):
            render_vector(paths[0], args)
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
