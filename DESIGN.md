# v2 design notes

## Geometry contract

The only free geometric parameters are the south-west ERA5 wave coordinate
`(lon0, lat0)`. Width and height are fixed at 0.5°. This makes all four wave
boundary controls exact ERA5 locations and makes the 0.25° wind-grid points
exactly the four edge midpoints plus the centre.

## Why the terrain path is U-Net-friendly

The SWAN input and Hs output are dense 2-D fields on the same rectangular grid.
The v2 tensor converter therefore keeps the terrain as a spatial channel and
maps the coarse forcing fields onto the same grid. A U-Net-style encoder/
decoder can use local bathymetry in the high-resolution path while using the
coarse bottleneck to represent domain-scale propagation. The first model
should remain deliberately small; increase width only after the 1-km pilot is
stable.

## Boundary caveat

GEBCO may place a rectangle corner on land. The shared SWAN engine snaps an
inactive ERA5 vertex to the nearest open contour within its configured snap
limit, and records that decision in `case_metadata.json` and
`boundary_forcing.csv`. A corner that is far from an open segment is a
configuration error, not something to silently repair in v2.
