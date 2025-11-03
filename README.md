# viewinline
[![Downloads](https://static.pepy.tech/badge/viewinline)](https://pepy.tech/project/viewinline)
[![PyPI version](https://img.shields.io/pypi/v/viewinline)](https://pypi.org/project/viewinline/)
[![Python version](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://pypi.org/project/viewinline/)

**Quick-look geospatial viewer for iTerm2.**  
Displays rasters and vectors directly in the terminal - no GUI, no temporary files.

This tool combines the core display logic of `viewtif` and `viewgeom`, but is **non-interactive**:  
you can’t zoom, pan, or switch colormaps on the fly. Instead, you control everything through command-line options (e.g. --display, --color-by, --colormap).

```bash
viewinline path/to/file.tif
viewinline path/to/vector.geojson
viewinline R.tif G.tif B.tif   # RGB composite
```
It’s designed for iTerm2 on macOS, using its inline image protocol to render a preview.
## Features  
- Displays rasters and vectors directly in the terminal  
- Works with iTerm2 inline image protocol 
- Non interactive: everything is controlled through command line options  

---

## Supported formats  
**Rasters**  
- GeoTIFF (`.tif`, `.tiff`)  
- Single band or multi band composites  

**Vectors**  
- GeoJSON (`.geojson`)  
- Shapefile (`.shp`, `.dbf`, `.shx`)  
- GeoPackage (`.gpkg`)  

**Composite inputs**  
- You can pass three rasters (e.g. `R.tif G.tif B.tif`) to create an RGB composite  

---
## Installation  
Requires Python 3.9 or later.  

```bash
pip install viewinline
```
### Available options
```bash
--display DISPLAY       # resize the displayed image (0.5=smaller, 2=bigger). default: auto fit to terminal
--ansi-size ANSI_SIZE   # set resolution if you are viewing the ANSI preview (try 180x90 or 200x100)
--band BAND             # band number to display (single raster case). default: 1
--colormap [{viridis,inferno,magma,plasma,cividis,terrain,RdYlGn,coolwarm,Spectral,cubehelix,tab10,turbo}]
                        # apply a colormap to single band rasters or vector coloring
                        # flag without value uses 'terrain' by default
--color-by COLOR_BY     # select a numeric column to color vector features
--edgecolor EDGECOLOR   # edge color for vector outlines (hex or named color). default: #F6FF00
--layer LAYER           # layer name for GeoPackage or multi layer files
```

### ANSI/ASCII color preview
If iTerm2 isn’t available, viewinline will automatically switch to an
ANSI/ASCII color preview or save a quick PNG under /tmp/viewinline_preview.png.

This mode works on terminals with **ANSI color support** and may not display correctly on others.  

For compatible terminals, `viewinline` renders images in a very coarse resolution. This feature is experimental.

## License
This project is released under the MIT License © 2025 Keiko Nomura.

If you find this tool useful, please consider supporting or acknowledging the author. 
