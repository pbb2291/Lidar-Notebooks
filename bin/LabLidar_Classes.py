# Defines Classes for Gridding/Voxelizing Lidar Data
# PB 
# 11/28/22
# Built on top of "Cloud_Class" used in the Buffalo Camp Paper
# and the Selenkay Diversity Project

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import time
import laspy
import concurrent.futures

# Point Cloud Class
# Used for voxelization
class Cloud: 
    
    def __init__(self,
                 lasf=None,
                 metrics={},
                 gridsize=1,
                 vsize=0.25,
                 heightcol='HeightAboveGround',
                 numretcol='number_of_returns',
                 retnumcol='return_number',
                 classcol='classification',
                 maxh=None): 
        
        self.lasf = lasf
        
        self.las = laspy.read(lasf)
        
        # initialize empty metrics dictionary
        self.metrics_dict = metrics
        
        # Set col names
        # NOTE: in future use self.las.point_format strings to regex these
        self.heightcol = heightcol
        self.numretcol = numretcol
        self.retnumcol = retnumcol
        self.classcol = classcol
        
        # Set grid and vert sizes
        self.gridsize = gridsize
        # Vertical res for foliage and cover profiles
        self.vsize = vsize

    # Make a grid with features defining each pixel
    def makegrid(self, xmin = None, xmax = None, ymin = None, ymax = None): 
        
        # If no grid boundaries given as input
        if not xmin:
            # Use the bounds of the lasfile to set bounds of the grid
            self.xmin, self.ymin, self.xmax, self.ymax = np.min(self.las.x), np.min(self.las.y), np.max(self.las.x), np.max(self.las.y)
        else:
            # else, use the boundaries of the input grid
            self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        
        # Build a grid of cells
        # by using floor to snap all x and y coords to a grid
        # divide all x and y values by cell size,
        # Subtract boundary coords (minx, miny) to snap to grid
        # floor them to get vertex coordinate of cell (lower left)
        # then multiply by size again and add mins to turn back into UTM coords
        x_vert = np.floor((self.las.x - self.xmin)/self.gridsize)*self.gridsize + self.xmin
        y_vert = np.floor((self.las.y - self.ymin)/self.gridsize)*self.gridsize + self.ymin

        # use np.unique to find unique combinations of x and y 
        # and record their indices (for later operations)
        # can take a bit, but took less than 30 sec for 15 million points with 5m grid size
        # Numpy unique docs:
        # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        xy_unique, idx = np.unique([x_vert, y_vert], axis=1, return_inverse=True)

        # unique indices for each cell 
        idx_unique = np.unique(idx)
        
        # output a dictionary
        self.grid_dict = {'x_cells': xy_unique[0],
                          'y_cells': xy_unique[1],
                          'idx_cells': idx_unique,
                          'idx_points': idx
                         }

        # NOTE: To run a function over all cells
        # just need to use idx_cells and idx_points in a loop:
        # for idx in grid_dict['idx_cells']: 
        #    cellsubset = las.points[idx_points == idx]
        #    do function(cellsubset)

        
# Cross-section class for plotting sections of a point cloud
class XSection:

    def __init__(self,
                 proj=None,
                 lasdir=None,
                 centerxy=None,
                 xsize=0.5,
                 ysize=30,
                 res=0.5,
                 points=[],
                 quantiles=[0.25, 0.5, 0.75, 1],
                 outdir=None):

        # Initialize attributes
        self.proj = proj
        self.centerxy = centerxy
        self.xsize = xsize
        self.ysize = ysize
        self.points = points
        self.res = res
        self.quantiles = quantiles
        self.outdir = outdir
                                           
        # Set outdirectory for figures (if you use save below)
        # Note: the plot function asks for this if it's empty and you try to save)
        self.outdir = outdir

        # Check if it's a project object being fed in
        if isinstance(self.proj, Classes.Project):
            
            # if so, grab some useful attributes
            self.projstr = self.proj.projstr
            self.lasdir = self.proj.lasdir

            # If no center for the XS was defined
            if centerxy is None:
                # set it as the center of the given project
                self.centerxy = proj.DTM['010m'].centercoords

        # if the folder of las directories exists
        if not self.lasdir is None:
            
            # make it an absolute path
            self.lasdir = os.path.abspath(self.lasdir)
            # and go fetch the las tiles you need
            self.getlastiles()
            # and go fetch the points you need from each tile
            self.getpoints()
            # Compute height above ground
            self.normheights()
            # compute percentile heights
            self.heightpercentiles()

    def getlastiles(self):

        # initialize list of lasfiles
        self.lasfiles = []
        self.headers = []
        
        # for each las file in the directory
        for f in glob.glob(self.lasdir + '/*.las'):
            # open the las file for reading
            with laspy.open(f) as l:
                
                # if the XS is long in the x direction
                if self.xsize >= self.ysize:
                    
                    # use the header and the min/max of the Xsection to check if the cross section falls in that tile
                    if ((((self.centerxy[0] + self.xsize/2) < l.header.maxs[0]) & ((self.centerxy[0] + self.xsize/2) > l.header.mins[0]) &
                          (self.centerxy[1] < l.header.maxs[1]) & (self.centerxy[1] > l.header.mins[1]) ) |
                        (((self.centerxy[0] - self.xsize/2) < l.header.maxs[0]) & ((self.centerxy[0] - self.xsize/2) > l.header.mins[0]) &
                            ((self.centerxy[1]) < l.header.maxs[1]) & (self.centerxy[1] > l.header.mins[1]))):
                        self.lasfiles.append(f)
                        self.headers.append(l.header)
                        
                else:
                    
                    # use the header and the min/max of the Xsection to check if the cross section falls in that tile
                    if ((((self.centerxy[1] + self.ysize/2) < l.header.maxs[1]) & ((self.centerxy[1] + self.ysize/2) > l.header.mins[1]) &
                          (self.centerxy[0] < l.header.maxs[0]) & (self.centerxy[0] > l.header.mins[0]) ) |
                        (((self.centerxy[1] - self.ysize/2) < l.header.maxs[1]) & ((self.centerxy[1] - self.ysize/2) > l.header.mins[1]) &
                            ((self.centerxy[0]) < l.header.maxs[0]) & (self.centerxy[0] > l.header.mins[0]))):
                        self.lasfiles.append(f)
                        self.headers.append(l.header)

    def getpoints(self):
        
        self.las_x_scaled = []
        self.las_y_scaled = []
        self.las_z_scaled = []
        self.classification = []
        
        # open up the las files
        for f in self.lasfiles:

            las = laspy.read(f)

            # subset the points to only default and ground points within the cross section
            # NOTE: 
            self.points = las.points[(las.x <= self.centerxy[0] + self.xsize/2) &
                                     (las.x >= self.centerxy[0] - self.xsize/2) &
                                     (las.y <= self.centerxy[1] + self.ysize/2) &
                                     (las.y >= self.centerxy[1] - self.ysize/2) &
                                     (las.classification != 7)]

             # Scale x, y, z and ground points
            self.las_x_scaled.append(self.points.array['X'] * self.headers[0].x_scale + self.headers[0].x_offset)
            self.las_y_scaled.append(self.points.array['Y'] * self.headers[0].y_scale + self.headers[0].y_offset)
            self.las_z_scaled.append(self.points.array['Z'] * self.headers[0].z_scale + self.headers[0].z_offset)

            self.classification.append(self.points.classification)
            
        # Concatenate (in the case of multiple files being selected)
        if len(self.classification) > 1:
            self.classification = np.concatenate(self.classification)
            self.las_x_scaled = np.concatenate(self.las_x_scaled)
            self.las_y_scaled = np.concatenate(self.las_y_scaled)
            self.las_z_scaled = np.concatenate(self.las_z_scaled)
        else:
            #otherwise, just grab them out of the list (they are numpy arrays)
            self.classification = self.classification[0]
            self.las_x_scaled = self.las_x_scaled[0]
            self.las_y_scaled = self.las_y_scaled[0]
            self.las_z_scaled = self.las_z_scaled[0]
            
        
        # Make a set of ground points
        self.ground_x_scaled = self.las_x_scaled[self.classification==2]
        self.ground_y_scaled = self.las_y_scaled[self.classification==2]
        self.ground_z_scaled = self.las_z_scaled[self.classification==2]

    def plotpoints(self, norm=False, returnfigandax=True, topdown=False, color=None, clabel='Class', fig=None, ax=None, colorbar=True):
        
        # if no figure is provided, make one
        if not fig:
            fig, ax = plt.subplots()

        if color is None:
            color = self.classification

        if topdown==True:
            a = ax.scatter(x=self.las_x_scaled, y=self.las_y_scaled,
                           c=color,
                           s=2, cmap='viridis', alpha=0.6)
            ax.axis('scaled')
            if colorbar==True:
                fig.colorbar(a, ax=ax, label=clabel)
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')

        else:
            if norm==True:
                try:
                    z = self.las_z_norm
                except:
                    self.normheights()
                    z = self.las_z_norm

            else:
                z = self.las_z_scaled

            if self.xsize >= self.ysize:
                a = ax.scatter(self.las_x_scaled, z, s=2, c=color, cmap='viridis', alpha=0.6)
                if colorbar==True:
                    fig.colorbar(a, ax=ax, label=clabel)
                ax.set_xlabel('Easting [m]')
                ax.axis('scaled')
                ax.set(ylim=(-0.2, np.max(z)+1))

            else:
                a = ax.scatter(self.las_y_scaled, z, s=2, c=color, cmap='viridis', alpha=0.6)
                if colorbar==True:
                    fig.colorbar(a, ax=ax, label=clabel)
                ax.set_xlabel('Northing [m]')
                ax.axis('equal')
                ax.set(ylim=(-0.2, np.max(z)+1))

        if returnfigandax==True:
            return fig, ax

        # plt.tight_layout()

    def makeshape(self):
        print("TBD \n")

        # Make a geojson polygon to clip a raster
        # Geojson formatting: https://python-geojson.readthedocs.io/en/latest/#coords
        # NOTE: This assumes a UTM projection
        # self.shape_json = geojson.Polygon([[(self.centerxy[0] + self.xsize/2, self.centerxy[1] + self.ysize/2),
        #                                    self.centerxy[0] - self.xsize/2,
        #                                    self.centerxy[1] + self.ysize/2,
        #                                    self.centerxy[1] - self.ysize/2]])

    def normheights(self, method='DTM'):
        # print("TBD")

        # using nearest:
        if method=='nearest':

            # use nearest ground point to norm heights
            self.las_z_norm = []
            # for each point
            for x, y, z in zip(self.las_x_scaled,
                               self.las_y_scaled,
                               self.las_z_scaled):

                # find the nearest ground point
                self.diff = (x - self.ground_x_scaled)**2 + \
                            (y - self.ground_y_scaled)**2 + \
                            (z - self.ground_z_scaled)**2

                idxmin = np.argmin(self.diff)

                # subtract the current ground z value
                self.las_z_norm.append(z - self.las_z_scaled[idxmin])

        # using DTM
        if method == 'DTM':

            # load the 10 cm DTM (replace with project points)
            DTM = rio.open(self.proj.DTM['010m'].filepath,
                           masked=True)

            # make a coordinate list of tuples from x and y
            # https: // geopandas.readthedocs.io / en / latest / gallery / geopandas_rasterio_sample.html
            coord_list = [(x, y) for x, y in zip(self.las_x_scaled, self.las_y_scaled)]

            # NOTE: this is a generator object, which prevents you from loading everything into memory
            # if you really want to use this right, iterate through it in the loop below instead
            # https://realpython.com/introduction-to-python-generators/
            DTM_elev = rio.sample.sample_gen(DTM,
                                             xy=coord_list,
                                             indexes=1,
                                             masked=False)

            # initialize list of heights
            self.las_z_norm = []

            # for each terrain elevation & point elevation
            for e, z, c in zip(DTM_elev, self.las_z_scaled, self.classification):
                # if it's a ground point
                if c == 2:
                    # set it to height 0
                    self.las_z_norm.append(0)

                else:
                    # subtract the point's elevation from the terrain to get height above ground
                    self.las_z_norm.append(z - e[0])

    def heightpercentiles(self):
        # print('TBD')

        # normalize z to height if not already done
        if not hasattr(self, 'las_z_norm'):
            self.normheights()

        # make a dataframe
        df = pd.DataFrame(data={'x': self.las_x_scaled,
                                'y': self.las_y_scaled,
                                'z': self.las_z_norm})

        # if the transect goes along the x (Easting) direction
        if self.xsize >= self.ysize:
            # use x to bin
            var = df.x

        else:
            # use y
            var = df.y

        # Bin by x coordinate
        # https://stackoverflow.com/questions/16947336/binning-a-dataframe-in-pandas-in-python/
        # generate correct number of bins for given res
        self.nbins = int(np.ceil((var.max() - var.min()) / self.res) + 1)

        # make bin edges
        self.binedges = np.linspace(var.min(), var.max(), num=self.nbins)

        # also, make bin centers (for plotting)
        self.bincenters = self.binedges + self.res/2

        # group by bin
        self.df_group = df.groupby(np.digitize(var, self.binedges))

        # Calc percentiles for each bin, and use "unstack" to flip output
        self.percentiles = pd.DataFrame(self.df_group.quantile(self.quantiles).z).unstack().to_dict('list')

        # Check for empty bins (empty areas without points in them)
        # will show if the total number of groups (number of group keys)
        # is less than the max value of the group ids
        if len(self.df_group.groups.keys()) != list(self.df_group.groups.keys())[-1]:

            print('Found empty bins. Filling... \n')

            # Make group id col for percentile dict
            bin_id = list(self.df_group.groups.keys())

            # for each potential group number
            for g in list(np.arange(1, self.nbins)):

                # check if the group is included, and if not
                if g not in bin_id:

                    # insert a group with 0s for each col in percentile dict
                    for k in self.percentiles.keys():
                        # insert a 0 value at the index of the bin
                        # Note: the bin index is the bin number - 1 (since python is 0 indexed)
                        self.percentiles[k].insert(g - 1, 0)


    def plotpercentiles(self, plotpoints=True, returnfigandax=True, fill=False, savefig=False, fig=None, ax=None, colorbar=True, fillcolor=None):

        if plotpoints == True:
            # if no figure provided
            if not fig:
                fig, ax = self.plotpoints(norm=True,
                                          topdown=False,
                                          color=self.las_z_norm,
                                          clabel='Height [m]',
                                          colorbar=colorbar)
            else:
                self.plotpoints(norm=True,
                                topdown=False,
                                color=self.las_z_norm,
                                clabel='Height [m]',
                                fig=fig, ax=ax,
                                colorbar=colorbar)
        else:
            # else, if there's not a fig provided, make one
            if not fig:
                fig, ax = plt.subplots()

        if fill == False:

            # loop through percentiles and plot each column
            for col in self.percentiles.keys():
                b = ax.plot(self.bincenters,
                            list(self.percentiles[col]),
                            '--',
                            label=f'RH{str(int(col[1]*100))}',
                            alpha=0.8)
                ax.legend()
        else:

            # convert into an array
            self.perc_array = np.array(list(self.percentiles.values()))

            # Make square bin edges (binedges at both sides of the res) for plotting
            self.binedges_square = []
            for left, right in zip(self.binedges, self.binedges + self.res):
                self.binedges_square.append(left)
                self.binedges_square.append(right)

            # loop through percentiles and plot each column
            for col in np.arange(0, self.perc_array.shape[0]):
                # If it's the first column of RHs
                if col == 0:
                    # Set the bottom of the plot fill to be 0 m (ground)
                    minvals = np.zeros(self.perc_array.shape[1])
                else:
                    # Else, use the next lowest RH as the min fill values
                    minvals = self.perc_array[col - 1]

                maxvals = list(self.perc_array[col])

                # duplicate values to match square bins
                minvals_square = []
                maxvals_square = []
                for minv, maxv in zip(minvals, maxvals):
                    # append each value twice to duplicate
                    minvals_square.append(minv)
                    minvals_square.append(minv)
                    maxvals_square.append(maxv)
                    maxvals_square.append(maxv)

                # Line plot (innacurate, but looks good)
                # b = ax.fill_between(self.bincenters,
                #                     minvals,
                #                     maxvals,
                #                     '--',
                #                     label=f'RH{str(int(list(self.percentiles.keys())[col][1] * 100))}',
                #                     alpha=0.3)
                
                # if there's no fillcolor, make it blue
                if not fillcolor:
                    fillcolor='C0'
                
                # Bar plot (accuracte), but looks less good)
                b = ax.fill_between(x=np.array(self.binedges_square),
                                    y2=np.array(minvals_square),
                                    y1=np.array(maxvals_square),
                                    label=f'RH{str(int(list(self.percentiles.keys())[col][1] * 100))}',
                                    alpha=0.25, color=fillcolor)
                ax.legend()

        # Print to check sizes
         # print(f'maxvals: {len(maxvals)} \n minvals: {len(minvals)} \n bincenters {len(self.bincenters)}')

        if savefig==True:
            # Check if there's an outdirectory, and ask user for input if not
            if not self.outdir:
                self.outdir = input('Specify directory path for saving figures: ')
                self.outdir = os.path.abspath(self.outdir)

            fig.savefig(self.outdir + f'/{self.proj.projstr}_{self.xsize}x{self.ysize}m_{self.res}res.png', dpi=300)

        if returnfigandax==True:
            return fig, ax