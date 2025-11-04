# viewinline
[![Downloads](https://static.pepy.tech/badge/viewinline)](https://pepy.tech/project/viewinline)
[![PyPI version](https://img.shields.io/pypi/v/viewinline)](https://pypi.org/project/viewinline/)
[![Python version](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://pypi.org/project/viewinline/)

**Quick-look geospatial viewer for iTerm2.**  
Displays rasters and vectors directly in the terminal - no GUI, no temporary files.

This tool combines the core display logic of `viewtif` and `viewgeom`, but is **non-interactive**: you can’t zoom, pan, or switch colormaps on the fly. Instead, you control everything through command-line options (e.g. --display, --color-by, --colormap).

It’s designed for iTerm2 on macOS, using its inline image protocol to render a preview.

## Installation  
Requires Python 3.9 or later.  

```bash
pip install viewinline
```
## Usage
```bash
viewinline path/to/file.tif
viewinline path/to/vector.geojson
viewinline R.tif G.tif B.tif                 # RGB composite
viewinline path/to/multiband.tif --rgb-bands 3,2,1
viewinline path/to/folder --gallery 4x3      # show image gallery (4x3 grid)
viewinline data.csv --describe               # show numeric summary
viewinline data.csv --hist                   # render inline histograms
viewinline data.csv --scatter X Y  # scatter plot
```

## Features  
- Displays rasters and vectors directly in the terminal  
- Works with iTerm2 inline image protocol 
- Non interactive: everything is controlled through command line options  

## Supported formats  
**Rasters**  
- GeoTIFF (.tif, .tiff)
- PNG, JPEG (.png, .jpg, .jpeg)
- Single-band or multi-band composites 

**Composite inputs**  
- You can pass three rasters (e.g. `R.tif G.tif B.tif`) to create an RGB composite  

**Vectors**  
- GeoJSON (`.geojson`)  
- Shapefile (`.shp`, `.dbf`, `.shx`)  
- GeoPackage (`.gpkg`)  

**CSV**
- Preview (--describe)
- Histograms (--hist)
- Scatter plots (--scatter x y [--marker dot|plus|x|square])

**Gallery view**
- Display all images in a folder with --gallery 4x4

### Available options
```bash
  --display DISPLAY     Resize only the displayed image (0.5=smaller, 2=bigger). Default: auto-fit to terminal.
  --ansi-size ANSI_SIZE
                        ANSI fallback resolution. Try 180x90 or 200x100.
  --band BAND           Band number to display (single raster case). (default: 1)
  --colormap [{viridis,inferno,magma,plasma,cividis,terrain,RdYlGn,coolwarm,Spectral,cubehelix,tab10,turbo}]
                        Apply colormap to single-band rasters or vector coloring. Flag without value → 'terrain'.
  --rgb-bands RGB_BANDS
                        Comma-separated band numbers for RGB display (e.g., '3,2,1'). Overrides default 1-3.
  --gallery [GRID]      Display all PNG/JPG/TIF images in a folder as thumbnails (e.g., 5x5 grid).
  --describe            Show numeric summary for CSV files (similar to pandas.describe). (default: False)
  --hist                Plot histograms for numeric columns in CSV. (default: False)
  --bins BINS           Number of bins for CSV histograms (used with --hist). (default: 20)
  --scatter X Y         Plot scatter of two numeric CSV columns (e.g. --scatter area_km2 year).
  --color-by COLOR_BY   Numeric column to color vector features by (optional).
  --width WIDTH         Line width for vector boundaries (default: 0.7)
  --edgecolor EDGECOLOR
                        Edge color for vector outlines (hex or named color). (default: #F6FF00)
  --layer LAYER         Layer name for GeoPackage or multi-layer files.
```

### ANSI/ASCII color preview
If iTerm2 isn’t available, viewinline will automatically switch to an
ANSI/ASCII color preview or save a quick PNG under /tmp/viewinline_preview.png.

This mode works on terminals with **ANSI color support** and may not display correctly on others.  

For compatible terminals, `viewinline` renders images in a very coarse resolution. This feature is experimental.

## License
This project is released under the MIT License © 2025 Keiko Nomura.

If you find this tool useful, please consider supporting or acknowledging the author. 
