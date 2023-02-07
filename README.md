# Lidar-Notebooks
A series of jupyter notebook pipelines for processing highly detailed lidar point clouds (LAS or LAZ files) and deriving vegetation structure metrics. 

### Processing Pipeline Scripts and What they do:

- LasFilePreprocessing
  A couple of tools for preprocessing las/laz point clouds.
    - *0-1-LasFiles_ComputeHeightClipBuffer* Computes the "HeightAboveGround" for each point using delauney triangulation of ground points. Also, removes buffer from the edge of a las tile (if specified). 
    - *0-2-LasBBoxShapefile* Creates shapefiles of bounding boxes of las files for context.
    
- PolygonMetrics
  A 2 part process for clipping las files with a set of polygons (1-ClipLasWithPolygons.ipynb) and then, drawing on las files to compute vegetation structure metrics for each polygon (2-ComputeMetricsByPolygon.ipynb). 
    - *1-ClipLasWithPolygons* - Clips las files using a set of polygon features, usually a large number of small plots (1-30 m).
    - *2-ComputeMetricsByPolygon* - Computes and save structural metrics for each polygon feature.
    
- VoxelMetrics
  A 3 part process for 1) clipping las files with a set of polygons (1-ClipLasWithPolygonsforVoxels.ipynb); 2) voxeling lidar data, computing vegetation structure metrics, and outputting a pickle file (2-ProcessVoxelMetrics.ipynb); and 3) outputting the pixel and voxel grids of each metric as geotif or netcdf files for use in qgis and other software (3-OutputVoxelMetrics_Geotiff_NetCDF.ipynb).
    - *1-ClipLasWithPolygonsforVoxels* - Clips las files using a set of polygon features, usually a small number of large plots (e.g. 1 ha)
    - *2-ProcessVoxelMetrics* - Voxelizes each clipped las file at the desired resolution, computes metrics for each voxel, and outputs pickle files.
    - *3-OutputVoxelMetrics_Geotiff_NetCDF* - Reads pickle files and outputs rasters as geotif files and voxel metrics as netcdf files for use in other GIS software.  
=======

