---
name: viewinline
description: Terminal viewer for rasters, vectors, and tabular data. Use for quick visual inspection without leaving the shell—preview images in a folder gallery, inspect geospatial files after GDAL workflows, explore CSV data with histograms/scatter plots, or verify files before commit. Provides machine-readable `--info` (metadata + statistics as JSON) and `--export` (save the rendered image) for programmatically checking generated geospatial files. Non-interactive; everything controlled via command-line flags.
tags: [visualization, terminal, raster, vector, csv, geospatial, gdal, ml, inspection, workflow]
---

# Viewinline Skill

Quick-look terminal viewer for geospatial and tabular data. Think of it as `ls` for visual files—designed for command-line workflows where you want to inspect data without leaving the terminal or opening a separate application.

Renders natively in iTerm2, WezTerm, Konsole, Rio, and Contour. Falls back to `chafa` (ASCII art with 24-bit color) in other terminals like kitty, Ghostty, Terminal.app, VS Code, and Linux terminals. Works over SSH without X11 forwarding or VNC.

## When to Use

- **File browsing**: `find . -name "*.jpg" | xargs -n1 viewinline` to visually scan results
- **Gallery view**: `viewinline path/to/folder --gallery 4x3` to preview all images in a directory
- **Band inspection**: Display specific bands of a raster as a grid: `--bands 10-50` or `--bands 11,15,30,45`
- **Geospatial workflows**: Inspect raster/vector outputs after GDAL transformations
- **ML training data**: Review training image galleries without switching windows
- **Data exploration**: Quick CSV visualizations (histograms, scatter plots, summary stats)
- **Pre-commit checks**: Verify images, maps, and data files before pushing
- **Remote servers**: Works over SSH from your local terminal
- **Verify generated files (agent workflows)**: After writing a GeoTIFF/NetCDF/vector with GDAL, rasterio, or xarray, run `viewinline FILE --info` to check the result is sensible (correct CRS, dimensions, value range, not all-NoData/NaN) as JSON, and `viewinline FILE --export out.png` to visually inspect it

## Supported Formats

**Rasters:** GeoTIFF, PNG, JPEG, NetCDF, HDF5, HDF4  
**Vectors:** GeoJSON, Shapefile, GeoPackage, GeoParquet  
**Tabular:** CSV, Parquet (with `pyarrow`)

## Inspecting Files (JSON output)

For programmatic checks, two flags return machine-readable output instead of drawing an image. Useful after generating a file to confirm it's sensible before continuing.

- `--info` — print file metadata and statistics as JSON, then exit. Reports facts, not judgments (the caller interprets them).
  - Rasters (GeoTIFF, NetCDF, HDF): format, dimensions, bands, dtype, CRS, resolution, bounds, nodata, per-band stats (min/max/mean, valid_fraction, naninf_fraction).
  - Vectors (GeoJSON, Shapefile, GeoPackage, GeoParquet): feature count, geometry type, CRS, bounds, columns.
  - For NetCDF/HDF: `--info` lists variables/subdatasets; add `--subset N` to inspect one.
  - Always returns JSON, including `{"readable": false, "error": ...}` on unreadable input — safe to parse in all cases.
- `--export PATH` — save the rendered image to PATH (`.png`/`.jpg`) and print `{"path": "..."}`. Works with any display flag (`--rgb`, `--colormap`, `--band`, `--display`), saving exactly what would be drawn.

```bash
# check a raster you just wrote
viewinline result.tif --info

# check a vector
viewinline boundaries.geojson --info

# NetCDF: list variables, then inspect one
viewinline data.nc --info
viewinline data.nc --subset 7 --info

# save a quick-look image to inspect visually
viewinline result.tif --export out.png
viewinline scene.tif --rgb 4 3 2 --export rgb.png
```

Note: `--info` and `--export` report what a file *is* and what it *looks like* — not whether the scientific result is correct.

## Core Flags

**Inspection (JSON output, no image):**
- `--info` — print metadata + statistics as JSON, then exit (rasters and vectors; lists variables for NetCDF/HDF with `--subset N`)
- `--export PATH` — save the rendered image to PATH (`.png`/`.jpg`); prints `{"path": "..."}`
- `--grid [N]` — overlay axis ticks and a light grid (N divisions, default 5). Labels are in the file's CRS units for georeferenced rasters and vectors, pixel positions otherwise. Combine with `--export` to save a quick-look with readable coordinates, e.g. to check a file covers the expected area.

**Raster Display:**
- `--rgb R G B` — Specify band order for RGB (e.g., `--rgb 4 3 2`)
- `--rgbfiles R.tif G.tif B.tif` — Create RGB composite from separate files
- `--band N` — Display specific band or NetCDF timestep
- `--bands RANGE` — Display multiple bands as a grid (e.g., `10-50` or `11,15,30,45`)
- `--gallery NxM` — Show all images in folder as thumbnails (e.g., `4x3` grid)
- `--colormap NAME` — Apply colormap (e.g., `plasma`, `viridis`, `terrain`)
- `--vmin VAL --vmax VAL` — Set min/max for scaling
- `--display SCALE` — Resize displayed output (0.5=smaller, 2=bigger)

**Vector Display:**
- `--color-by COLUMN` — Color features by attribute value
- `--colormap NAME` — Apply colormap to coloring
- `--width W` — Line width for boundaries
- `--edgecolor COLOR` — Edge color (hex or named)
- `--table` — Display as tabular data instead of rendering geometry

**Tabular Data (CSV/Parquet):**
- `--describe [COL]` — Summary statistics for all numeric columns or one
- `--hist [COL]` — Histograms for numeric columns
- `--scatter X Y` — Scatter plot of two columns
- `--where EXPR` — Filter rows (DuckDB required): `--where "year > 2010"`
- `--sort COL` — Sort by column (ascending); use `--desc` for descending
- `--limit N` — Limit output rows
- `--select COL1 COL2` — Choose specific columns to display
- `--sql QUERY` — Full DuckDB SQL (use `data` as table name)

**NetCDF/HDF:**
- `--subset N` — Select variable by index
- `--band N` or `--timestep N` — Select slice along time/band axis
- `--reduce DIM_NAME` — Override auto-detected band axis for non-standard dimensions

## Examples

### Image browsing
```bash
# Browse all JPGs in directory
find . -name "*.jpg" | xargs viewinline

# Gallery of all images in a folder (4 columns × 3 rows)
viewinline path/to/images --gallery 4x3
```

### Raster workflows
```bash
# Display single band
viewinline data.tif --band 2

# RGB composite from three bands
viewinline multiband.tif --rgb 4 3 2

# Create RGB from separate files
viewinline R.tif G.tif B.tif

# Display band range as gallery
viewinline hyperspectral.tif --bands 10-50 --gallery 5x5

# Apply colormap with min/max scaling
viewinline temp.nc --subset 1 --colormap plasma --vmin 273 --vmax 310
```

### GDAL verification
```bash
# Reproject and check result
gdalwarp -t_srs EPSG:3857 input.tif temp.tif
viewinline temp.tif --colormap terrain

# Resample and verify
gdal_translate -outsize 50% 50% temp.tif output.tif
viewinline output.tif
```

### Vector inspection
```bash
# View vector file
viewinline boundaries.geojson

# Color features by attribute, with colormap
viewinline boundaries.geoparquet --color-by population --colormap viridis

# View vector as tabular data
viewinline counties.shp --table
viewinline data.geoparquet --table --where "POP > 100000" --sort POP --desc
```

### Data exploration
```bash
# Preview CSV
viewinline data.csv

# Summary statistics
viewinline data.parquet --describe

# Histograms for all numeric columns
viewinline data.csv --hist

# Scatter plot
viewinline data.csv --scatter area_km2 year

# Filter and sort
viewinline data.csv --where "year > 2010" --sort population --desc

# SQL query
viewinline data.csv --sql "SELECT * FROM data WHERE area > 100 ORDER BY year"

# Batch CSV inspection with xargs (use -I {} to control argument order)
find . -name "*.csv" | xargs -n1 -I {} viewinline {} --describe
```

### Verify generated files (agent workflows)
```bash
# Reproject, then check the result is sensible as JSON
gdalwarp -t_srs EPSG:3857 input.tif output.tif
viewinline output.tif --info

# Confirm a written raster isn't all-NoData / has expected CRS and range
viewinline result.tif --info

# List a NetCDF's variables, then inspect the data variable
viewinline output.nc --info
viewinline output.nc --subset 7 --info

# Save a quick-look PNG to inspect visually
viewinline result.tif --export check.png
```

## Tips

- Use `--gallery` to quickly preview batches of training images or search results
- Pipe with `xargs`: `find . -newer file.txt | xargs viewinline` to inspect recent changes
- Add aliases: `alias check='viewinline --describe --hist'`
- Works in tmux if outer terminal is iTerm2; falls back to ASCII art in other terminals
- Install `chafa` for better coverage: `brew install chafa` (macOS), `apt install chafa` (Linux)
- SSH-friendly: images render on your local terminal, not the remote server
