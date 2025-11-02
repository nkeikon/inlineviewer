# viewinline
**Quick-look geospatial viewer for iTerm2.**  
Displays rasters and vectors directly in the terminal — no GUI, no temporary files.

This tool combines the core display logic of `viewtif` and `viewgeom`, but is **non-interactive**:  
you can’t zoom, pan, or switch colormaps on the fly. Instead, you control everything through command-line options (e.g. --display, --color-by, --colormap).

```bash
viewinline path/to/file.tif
viewinline path/to/vector.geojson
viewinline R.tif G.tif B.tif   # RGB composite
```
It’s designed for iTerm2 on macOS, using its inline image protocol to render a preview.

### Available options
```bash
### Common options
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
### Dependencies
Requires Python 3.9 or later and the following libraries:
```bash
pip install numpy pillow rasterio geopandas matplotlib pyogrio
```

### ANSI/ASCII color preview
If iTerm2 isn’t available, viewinline will automatically switch an
ANSI/ASCII color preview or save a quick PNG under /tmp/viewinline_preview.png.

This mode works on terminals with **ANSI color support** and may not display correctly on others.  
For compatible terminals, `viewinline` renders images in a very coarse resolution. This feature is experimental.
# inlineviewer
