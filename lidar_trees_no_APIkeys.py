from multiprocessing import Pool
from osgeo import gdal
from pyproj import Proj, transform
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm

import datetime
import fiona
import geojson
import geopandas as gpd
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdal
import rasterio
import rasterio.mask
import requests
import scipy.spatial
import shapely
import shapely.geometry
import sys
import urllib.request

sys.path.append('src')
from pcl_utils import local_max

# Program variables

output_folder = 'output'
NEARMAP_API_KEY = ''
PLANET_API_KEY = ''
las_shapefile = 'RBG_LAS_data.shp'
boundary_shapefile = 'RBG.shp'
front_map = 'display_photo.png'

''' Load predefined boundary of the RBG from a shapefile, and get bounding
    box GeoJSON for Planet API. Ideally, after importing all the las files,
    a bounding box should emcompass all necessary areas for Planet data.
    As we know already know all the data is inside the RBG, this is the
    quickest implementation as I already had this as a shapefile.
'''

RBG_shapefile = gpd.read_file(boundary_shapefile)
RBG_shapefile = RBG_shapefile.to_crs(epsg=4326)
RBG_bbox = json.loads(RBG_shapefile.boundary.to_json())['features'][0]['bbox']
RBG_bbox = shapely.geometry.box(
    RBG_bbox[0], RBG_bbox[1], RBG_bbox[2], RBG_bbox[3])
RBG_boundary = geojson.Polygon(RBG_bbox.exterior.coords)['coordinates']


# ## LiDAR functions

def import_lidar(lidar_filename):

    lidar_data = {
        "pipeline": [lidar_filename,
                     {"type": "filters.pmf"},
                     {"type": "filters.hag"},
                     {"type": "filters.eigenvalues", "knn": 16},
                     {"type": "filters.normal", "knn": 16},
                     {"type": "filters.outlier",
                         "method": "statistical",
                         "multiplier": 3,
                         "mean_k": 8
                      },
                     {
                         "type": "filters.range",
                         "limits": "Classification![7:7],Z[-100:3000]"
                     }
                     ]}
    pipeline = pdal.Pipeline(json.dumps(lidar_data))
    pipeline.validate()
    n_points = pipeline.execute()
    arr = pipeline.arrays[0]
    description = arr.dtype.descr
    cols = [col for col, __ in description]
    df = pd.DataFrame({col: arr[col] for col in cols})
    df['X_0'] = df['X']
    df['Y_0'] = df['Y']
    df['Z_0'] = df['Z']
    df['X'] = df['X'] - df['X_0'].min()
    df['Y'] = df['Y'] - df['Y_0'].min()
    df['Z'] = df['Z'] - df['Z_0'].min()

    # Some df columns based off of some form of logic (from tutorial)
    df['ground'] = df['Classification'] != 1
    df['flatness'] = df['Eigenvalue0']
    df['tree_potential'] = (df['Classification'] == 1) & (
        df['HeightAboveGround'] >= 0.5)
    df['other'] = ~df['ground'] & ~df['tree_potential']
    del(pipeline)
    del(n_points)
    del(arr)
    del(cols)
    return df


def euclid_dist(fromx, fromy, tox, toy):
    ''' A simple euclidean function 2D space,
        as tree locations are only 2D.
    '''
    delta_x = float(tox)-float(fromx)
    delta_y = float(toy)-float(fromy)
    return (math.sqrt((delta_x*delta_x)+(delta_y*delta_y)))


def process_the_lidar(filename):
    # Get today's date for Nearmap data imagery to pull the latest image
    todays_date = datetime.datetime.today().strftime('%Y%m%d')

    trees = gpd.read_file(las_shapefile)

    # Name the output as the RBG plant id number
    lidarextension = filename.split('/')[-1].split('.')[0][-2:]
    lidarnumber = str(trees['RBGno'].loc[trees['LASfile']
                                         == 'merged_0000'
                                         + str(lidarextension)].values[0])

    if os.path.isfile(os.path.join(os.getcwd(), output_folder,
                                   'lidar', lidarnumber+'meta.csv')):
        print(lidarnumber, '- Already has been run, skipping')
    else:
        df = import_lidar(filename)
        df = df.loc[df['tree_potential']]
        print(lidarnumber, "- Amount of tree points in dataframe:", len(df))

        lep = local_max(df.loc[df['tree_potential'],
                               ['X', 'Y', 'Z',
                                'HeightAboveGround',
                                'X_0', 'Y_0', 'Z_0']],
                        radius=3, density_threshold=15)

        # Cluster the trees using KDTree but something better could be used
        kdtree = scipy.spatial.kdtree.KDTree(lep[['X', 'Y', 'Z']])
        dist, idx = kdtree.query(
            df.loc[df['tree_potential'], ['X', 'Y', 'Z']].values)
        df.loc[df['tree_potential'], 'tree_idx'] = idx

        # Define centroid of XYZ plane
        centroid = df['Y'].mean(), df['Z'].mean(), df['X'].mean()

        # Calculate distance of tree tops to centroid, add distance to lep
        tree_distances = []
        for tree in lep.values.tolist():
            tree_distances.append(euclid_dist(
                tree[0], tree[1], centroid[0], centroid[1]))
        lep['distance'] = tree_distances

        # Get treetop with shortest dist to centroid. Its index now the tree id
        closest_treetop = lep[lep['distance'] == lep['distance'].min()]
        # Build DF corresponding to tree matched in closest_treetop search
        centre_tree_df = df.loc[df['tree_idx'] ==
                                df.loc[closest_treetop.index[0]]['tree_idx']]

        # Get CRS of project from  shapefile of trees (caution: not the lidar)
        project_crs = trees.crs['init']
        nearmap_crs = 'epsg:4326'

        # Reproject coordinate of tree to one for Nearmap (EPSG:4326)
        inProj = Proj(init=project_crs)
        outProj = Proj(init=nearmap_crs)
        x1, y1 = (closest_treetop[['X_0', 'Y_0']].values[0][0],
                  closest_treetop[['X_0', 'Y_0']].values[0][1])
        x2, y2 = transform(inProj, outProj, x1, y1)

        # Tree's coordinates in WGS84
        tree_coords_wgs84 = (x2, y2)
        tree_coords_wgs84utm = (x1, y1)
        print(lidarnumber, "- WGS84 UTM coordinates are:\t",
              str(tree_coords_wgs84utm[1])+','+str(tree_coords_wgs84utm[0]))
        print(lidarnumber, "- WGS84 long coordinates are:\t",
              str(tree_coords_wgs84[1])+','+str(tree_coords_wgs84[0]))

        if not os.path.isdir(os.path.join('.', output_folder)):
            os.makedirs(os.path.join('.', output_folder))
            print(lidarnumber, "- Created folder",
                  os.path.join('.', output_folder))
        if not os.path.isdir(os.path.join('.', output_folder, 'lidar')):
            os.makedirs(os.path.join('.', output_folder, 'lidar'))
            print(lidarnumber, "- Created folder",
                  os.path.join('.', output_folder, 'lidar'))

        # Volume and height of tree cloud. Assume CRS in mtr (UTM WGS84)
        tree_volume = round(scipy.spatial.ConvexHull(
            centre_tree_df[['X', 'Y', 'Z']]).volume, 2)
        tree_height = round(closest_treetop['HeightAboveGround'].values[0], 2)
        print(lidarnumber, "- Dimensions of the tree are:\t Volume:",
              str(tree_volume)+"m^3", "Height:", str(tree_height)+"m")

        # Save lidar 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.dist = 8
        ax.scatter(centre_tree_df['X'].values, centre_tree_df['Y'].values,
                   centre_tree_df['Z'].values, zdir="z", s=1, marker='^')
        ax.view_init(elev=10., azim=0)
        fig.savefig(os.path.join(os.getcwd(), output_folder,
                                 'lidar', lidarnumber+'_plot0deg.jpg'),
                    dpi=400)
        print("{} - Saved 0° plot to:\t\t {}".format(
            lidarnumber,
            os.path.join('.',
                         output_folder, 'lidar',
                         lidarnumber+'_plot0deg.jpg')))
        ax.view_init(elev=10., azim=45)
        fig.savefig(os.path.join(os.getcwd(),
                                 output_folder,
                                 'lidar',
                                 lidarnumber + '_plot45deg.jpg'), dpi=400)
        print("{} - Saved 45° plot to:\t\t{} ".format(
            lidarnumber, os.path.join('.', output_folder,
                                      'lidar',
                                      lidarnumber+'_plot45deg.jpg')))
        ax.view_init(elev=10., azim=90)
        fig.savefig(os.path.join(os.getcwd(), output_folder,
                                 'lidar',
                                 lidarnumber+'_plot90deg.jpg'), dpi=400)
        print("{} - Saved 90° plot to:\t\t{}".format(
            lidarnumber, os.path.join('.', output_folder,
                                      'lidar',
                                      lidarnumber+'_plot45deg.jpg')))
        plt.close()

        # Obtain nearmap imagery
        nearmap_url = "http://au0.nearmap.com/staticmap?center=" \
            + str(tree_coords_wgs84[1]) + "," + str(tree_coords_wgs84[0]) \
            + "&size=3000x3000&zoom=22&date=" \
            + todays_date+"&httpauth=false&apikey="+NEARMAP_API_KEY
        fn, _ = urllib.request.urlretrieve(nearmap_url, os.path.join(
            os.getcwd(), output_folder, 'lidar', lidarnumber+'_aerial.jpg'))
        print(lidarnumber, "- Saved Nearmap imagery to:\t",
              os.path.join('.', output_folder, 'lidar',
                           lidarnumber+'_aerial.jpg'))

        import csv
        with open(os.path.join(os.getcwd(), output_folder, 'lidar',
                               lidarnumber+'meta.csv'), 'w') as csvfile:
            lidar_metawriter = csv.writer(csvfile, delimiter=',')
            lidar_metawriter.writerow(
                ['treenumber', 'long', 'lat', 'utm_x', 'utm_y',
                 'height', 'tree_volume'])
            lidar_metawriter.writerow([lidarnumber, tree_coords_wgs84[0],
                                       tree_coords_wgs84[1],
                                       tree_coords_wgs84utm[0],
                                       tree_coords_wgs84utm[1],
                                       tree_height, tree_volume])
        print(lidarnumber, "- Written csv file: ./" +
              output_folder+"/lidar/"+lidarnumber+"meta.csv")


def run_lidar_multithread():
    pool = Pool()
    # Start the larger files first. Will finish quicker this way
    pool.map(process_the_lidar, sorted(glob.iglob(
        os.getcwd()+'/clipped/*.las'), key=os.path.getsize, reverse=True))
    pool.close()
    pool.join()


''' Planet functions

    This code has been adapted from the Planet API resource notebooks at:
    https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/data-api-tutorials/planet_data_api_introduction.ipynb
    as well as their NDVI method at:
    https://developers.planet.com/tutorials/calculate-ndvi/
'''


# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))

# Function to assist with downloading Planet data


def pl_download(url, folder, filename=None):
    # Send a GET request to the provided location url, using API Key
    res = requests.get(url, stream=True, auth=(PLANET_API_KEY, ""))
    # If no filename argument is given
    if not filename:
        # Construct a filename from the API response
        if "content-disposition" in res.headers:
            filename = res.headers["content-disposition"].split(
                "filename=")[-1].strip("'\"")
        # Construct a filename from the location url
        else:
            filename = url.split("=")[1][:10]

    if not os.path.isdir(os.path.join('.', folder)):
        os.makedirs(os.path.join('.', folder))
        print("Created folder", os.path.join('.', folder))
    if not os.path.isfile(os.path.join('.', folder, filename)):
        print("Downloading...")
        with open(os.path.join('.', folder, filename), "wb") as f:
            for chunk in res.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        print(os.path.join('.', folder, filename), "is downloaded")
        print("Creating NDVI and .png preview")
        form_ndvi(filename, folder)
    else:
        print(os.path.join('.', folder, filename), "already exists")

    return filename

# Takes the input features list and returns a list of date, id and cloud cover


def concat_feature_list(features, RBG_bbox):
    feature_list = []
    for f in features:
        # Build bbox with Shapely to check if it fits our project's geometry
        feature_bbox = shapely.geometry.Polygon(
            f['geometry']['coordinates'][0])
        date = f["properties"]['acquired'].split(
            ':')[0].split('T')[0].split('-')
        feature_list.append([date[2], date[1], date[0], f['id'],
                             f['properties']['cloud_cover'],
                             f['_links']['assets'],
                             f['properties']['sun_azimuth'],
                             f['properties']['sun_elevation'],
                             f['properties']['view_angle'],
                             feature_bbox.contains(RBG_bbox)])
    return feature_list


def form_ndvi(filename, folder):
    ''' The following NDVI calculation method is adapted from
        https://developers.planet.com/tutorials/calculate-ndvi/
    '''

    with rasterio.open(os.path.join('.', folder, filename)) as src:
        band_red = src.read(3)
    with rasterio.open(os.path.join('.', folder, filename)) as src:
        band_nir = src.read(4)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI
    ndvi = (band_nir.astype(float) - band_red.astype(float)) / \
        (band_nir + band_red)

    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1)

    # Create the file
    with rasterio.open(os.path.join('.', folder, filename+'.NDVI.tif'),
                       'w', **kwargs) as dst:
        dst.write_band(1, ndvi.astype(rasterio.float32))

    # Export as PNG
    plt.imsave(os.path.join('.', folder, filename+'ndvi_cmap.png'), ndvi)


def acquire_planet_data(feature_asset_url, output_folder, session):
    print("Working on", feature_asset_url)

    # Get the assets link for the item
    assets_url = feature_asset_url
    # Send a GET request to the assets url for the list of available assets
    res = session.get(assets_url)
    # Assign a variable to the response
    assets = res.json()
    # We want the analytic datatype
    analytic = assets["analytic"]

    # Setup the activation url for a particular asset (the visual asset)
    activation_url = analytic["_links"]["activate"]

    # Send a request to the activation url to activate the item
    res = session.get(activation_url)

    # Print the response from the activation request

    if res.status_code == 202:
        print("202 - The request has been accepted and the"
              "activation will begin shortly.")
    elif res.status_code == 204:
        print("204 - The asset is already active and no "
              "further action is needed.")
    elif res.status_code == 401:
        print("401 - The user does not have permissions to "
              "download this file.")
    else:
        print("Undefined error.")

    asset_activated = False

    while asset_activated is False:
        # Send a request to the item's assets url
        res = session.get(assets_url)

        # Assign a variable to the item's assets url response
        assets = res.json()

        # Assign a variable to the visual asset from the response
        analytic = assets["analytic"]

        asset_status = analytic["status"]

        # If asset is already active, we are done
        if asset_status == 'active':
            asset_activated = True
            print("Asset is active and ready to download")


def crop_ndvi(shapefile, rasterin, rasterout, folder):
    outfolder = os.path.join('.', folder, 'ndvi')

    # Create NDVI folder
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
        print("Created folder", outfolder)

    with fiona.open(shapefile, "r") as shapefile:
        shpfeatures = [feature["geometry"] for feature in shapefile]

    with rasterio.open(os.path.join('.', folder, rasterin), "r") as src:
        out_image, out_transform = rasterio.mask.mask(
            src, shpfeatures, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(os.path.join(outfolder, rasterout),
                       "w", **out_meta) as dest:
        dest.write(out_image)


def get_planet_data(output_folder):
    print("\n--- Obtaining Planet Data & Calculating NDVI ---")
    # Setup Planet Data API base URL
    URL = "https://api.planet.com/data/v1"

    # Setup the session
    session = requests.Session()

    # Authenticate
    session.auth = (PLANET_API_KEY, "")

    # Specify the sensors/satellites or "item types" to include in our results,
    # date range, cloud cover and bounding box
    item_types = ["PSScene4Band"]

    # Set from 2 years from today
    thisyear = datetime.datetime.today().strftime('%Y')
    thismonth = datetime.datetime.today().strftime('%m')
    thisday = datetime.datetime.today().strftime('%d')
    thistime = 'T00:00:00Z'
    planet_start = str(int(thisyear)-2)+'-'+thismonth+'-'+thisday+thistime
    planet_end = thisyear+'-'+thismonth+'-'+thisday+thistime

    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gt": planet_start,
            "lte": planet_end
        }
    }

    range_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lt": 0.1
        }
    }

    RBG_geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": {
            "type": "Polygon",
            "coordinates": [
                RBG_boundary
            ]
        }
    }

    # Combine the filters together in the config item
    and_filter = {
        "type": "AndFilter",
        "config": [range_filter, RBG_geometry_filter, date_filter]
    }

    # Make a GET request to the Planet Data API
    res = session.get(URL)

    # Setup the stats URL
    stats_url = "{}/stats".format(URL)

    # Construct the request.
    request = {
        "item_types": item_types,
        "interval": "month",
        "filter": and_filter
    }
    # Send the POST request to the API stats endpoint
    res = session.post(stats_url, json=request)

    # Setup the quick search endpoint url
    quick_url = "{}/quick-search".format(URL)

    request = {
        "item_types": item_types,
        "interval": "month",
        "filter": and_filter
    }

    # Send the POST request to the API quick search endpoint
    res = session.post(quick_url, json=request)

    # Assign the response to a variable
    geojson = res.json()
    features = geojson["features"]

    # Loop over all the features from the response
    suitable_images = pd.DataFrame(concat_feature_list(features, RBG_bbox),
                                   columns=['day', 'month', 'year', 'id',
                                            'cloud', 'asset_url',
                                            'sun_azimuth', 'sun_elevation',
                                            'view_angle', 'fits_project'])
    print("Features:", len(features), "Total features:", len(suitable_images))

    # If page reaches maximum, select next and concat results until all found
    while len(features) == 250:
        next_url = geojson["_links"]["_next"]
        res = session.get(next_url)
        geojson = res.json()
        features = geojson["features"]
        suitable_images = suitable_images.append(
            pd.DataFrame(concat_feature_list(features, RBG_bbox),
                         columns=['day', 'month', 'year', 'id',
                                  'cloud', 'asset_url', 'sun_azimuth',
                                  'sun_elevation', 'view_angle',
                                  'fits_project']))
        print("Features:", len(features),
              "Total features:", len(suitable_images))
    print("Finished loading results")

    # Reset the dataframe index after appending
    suitable_images = suitable_images.reset_index()

    # Clean out images that don't fit within the boundary of our project
    suitable_images = suitable_images.loc[
        suitable_images['fits_project'] == True]

    # Find day of each month over data series, where cloud cover is least

    df1 = suitable_images.loc[suitable_images.groupby(
        ['year', 'month'])['cloud'].idxmin()].set_index(['year', 'month'])
    df2 = suitable_images.loc[suitable_images.groupby(
        ['year', 'month'])['cloud'].idxmax()].set_index(['year', 'month'])

    unique_days = pd.concat([df1, df2], axis=1, keys=('min', 'max'))
    unique_days.columns = unique_days.columns.map('_'.join)

    # Should be ready to download... Go!
    for asset_url in unique_days['min_asset_url']:
        acquire_planet_data(asset_url, output_folder, session)

    satellite_data = unique_days
    satellite_data['year'], satellite_data['month'] = satellite_data.index[0]
    satellite_data = satellite_data[['year', 'month', 'min_day',
                                     'min_cloud', 'min_sun_azimuth',
                                     'min_sun_elevation', 'min_view_angle']]
    satellite_data = satellite_data.rename(
        columns={'min_day': 'day',
                 'min_cloud': 'cloud_cover',
                 'min_sun_azimuth': 'sun_azimuth',
                 'min_sun_elevation': 'sun_elevation',
                 'min_view_angle': 'view_angle'})

    return satellite_data


def crop_the_planet():
    print("\n------- Cropping Planet Data -------")
    for tif in glob.iglob(os.path.join(os.getcwd(), 'output', '*NDVI.tif')):
        outtif = os.path.join(
            tif.split('/')[-1:][0].split('_')[0][:-2] + '.ndvi.tif')
        crop_ndvi(boundary_shapefile, tif, outtif, output_folder)
        print(tif.split('/')[-1].split('_')[0][:-2], "- Cropped NDVI:", outtif)

    for tif in glob.iglob(os.path.join(os.getcwd(), 'output',
                                       'ndvi', '*.tif')):
        with rasterio.open(tif) as src:
            ndvi = src.read(1)
        png_filename = os.path.join(
            '.', 'output', 'ndvi', tif.split('.')[0]+'ndvi_cmap.png')
        plt.imsave(png_filename, ndvi)
        print(tif.split('/')[-1].split('.')[0], "- PNG file output:",
              png_filename.split(' ')[-1].split('/')[-1])


# ## Post processing functions

def import_lidar_csvs(output_folder):
    lidar_data = pd.DataFrame(
        columns=['treenumber', 'long', 'lat', 'utm_x', 'utm_y',
                 'height', 'tree_volume'])
    for file in glob.iglob(os.path.join(os.getcwd(), output_folder,
                                        'lidar', '*.csv')):
        csv_pd = pd.read_csv(file)
        lidar_data = pd.concat([csv_pd, lidar_data], sort=False)
    return lidar_data

# Function to return point at coordinate, adapted from
# https://gis.stackexchange.com/questions/221292/retrieve-pixel-value-with-geographic-coordinate-as-input-with-gdal


def ndvi_at_coordinate(filename, X, Y):
    filename = filename
    dataset = gdal.Open(filename)
    band = dataset.GetRasterBand(1)

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    transform = dataset.GetGeoTransform()

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    data = band.ReadAsArray(0, 0, cols, rows)

    point = X, Y  # list of X,Y coordinates

    col = int((point[0] - xOrigin) / pixelWidth)
    row = int((yOrigin - point[1]) / pixelHeight)

    return round(float(data[row][col]), 2)


def plot_lidar_data(lidar_data, output_folder):
    plot_title = "NDVI Index"
    for i in range(len(lidar_data)):
        plot = lidar_data.iloc[i].filter(regex='20').plot(
            figsize=(20, 5),
            title="Tree: "
            + str(lidar_data.iloc[i]['treenumber'])
            + "'s " + plot_title)
        plt.xticks(np.arange(25), lidar_data.iloc[i].index[7:])
        fig = plot.get_figure()
        fig.savefig(os.path.join(
            os.getcwd(), './', output_folder,
            'lidar', str(lidar_data.iloc[i][0])+"_ndvi_plot.png"))
        plt.close()


def output_pdf(lidar_data, pdf_filename):
    # Join two shapefiles together for extracted lidar data and tree info
    tree_data = gpd.read_file(las_shapefile)
    tree_data = tree_data.set_index('RBGno').join(
        lidar_data.set_index('treenumber'), how='inner')

    # Create PDF. A lot of this is manual, especially the begining
    c = canvas.Canvas(os.path.join(os.getcwd(), pdf_filename))

    c.setFillColor('black')
    c.setFont("Helvetica-Bold", 20)
    c.drawString(175, 750, "Royal Botanic Gardens")
    c.drawImage(os.path.join(os.getcwd(), front_map), 100, 300, 15*cm, 15*cm)
    c.setFont("Helvetica", 14)
    c.drawString(175, 250, "LiDAR Acquisition date: 21-4-2017")
    c.drawString(175, 230, "Nearmap latest imagery as of: " +
                 str(datetime.datetime.today().strftime('%d-%m-%Y')))
    c.drawString(175, 210, "Planet data series from: " +
                 str(datetime.datetime.today().strftime('10-2016')))
    c.drawString(175, 190, "Planet data series until: " +
                 str(datetime.datetime.today().strftime('10-2018')))
    c.drawString(175, 150, "Authored by: Evan Thomas, 705935")
    c.drawString(175, 130, "Subject: Remote Sensing GEOM90005")
    # move the origin up and to the left
    c.translate(inch, inch)

    # choose some colors
    c.setStrokeColor('0.2,0.5,0.3')
    c.setFillColor('green')
    # draw some lines
    c.line(-50, -50, -50, 3.6*cm)
    c.line(-50, -50, -50+(3.2*cm), -50)
    # draw a rectangle
    c.rect(-50+(0.5*cm), -50+(0.5*cm), -50+(4.2*cm), -50+(5.2*cm), fill=1)
    # Lower corner icon
    # make text go straight up
    c.rotate(90)
    # change color
    c.setFillColor('yellow')
    # say hello (note after rotate the y coord needs to be negative!)
    # define a large font
    c.setFont("Helvetica", 14)
    c.drawString(-50+(0.7*cm), 50+(-2*cm), "TreeGIS")
    c.drawString(-50+(0.7*cm), 50+(-2.5*cm), "Consultants")
    c.showPage()

    # Show the series of NVDI images
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201610ndvi_cmap.png'), 20, 650, 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201611ndvi_cmap.png'), 20+(4*cm), 650, 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201612ndvi_cmap.png'), 20+(8*cm), 650, 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201701ndvi_cmap.png'), 20+(12*cm), 650, 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201702ndvi_cmap.png'), 20+(16*cm), 650, 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201703ndvi_cmap.png'), 20, 650-(4*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201704ndvi_cmap.png'), 20+(4*cm), 650-(4*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201705ndvi_cmap.png'), 20+(8*cm), 650-(4*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201706ndvi_cmap.png'), 20+(12*cm), 650-(4*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201707ndvi_cmap.png'), 20+(16*cm), 650-(4*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201708ndvi_cmap.png'), 20, 650-(8*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201709ndvi_cmap.png'), 20+(4*cm), 650-(8*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201710ndvi_cmap.png'), 20+(8*cm), 650-(8*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201711ndvi_cmap.png'), 20+(12*cm), 650-(8*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201712ndvi_cmap.png'), 20+(16*cm), 650-(8*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201801ndvi_cmap.png'), 20, 650-(12*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201802ndvi_cmap.png'), 20+(4*cm), 650-(12*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201803ndvi_cmap.png'), 20+(8*cm), 650-(12*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201804ndvi_cmap.png'), 20+(12*cm), 650-(12*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201805ndvi_cmap.png'), 20+(16*cm), 650-(12*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201806ndvi_cmap.png'), 20, 650-(16*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201807ndvi_cmap.png'), 20+(4*cm), 650-(16*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201808ndvi_cmap.png'), 20+(8*cm), 650-(16*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201809ndvi_cmap.png'), 20+(12*cm), 650-(16*cm), 3.8*cm, 3.8*cm)
    c.drawImage(os.path.join(
        os.getcwd(), 'output', 'ndvi',
        '201810ndvi_cmap.png'), 20+(16*cm), 650-(16*cm), 3.8*cm, 3.8*cm)

    c.drawString(140, 150, "Planet (PlanetScope4) data")
    c.drawString(
        140, 130, "Applying Normalised Difference Vegetation Index (NDVI)")
    c.drawString(140, 110, "Dates: 10/2016, to 10/2018")

    # move the origin up and to the left
    c.translate(inch, inch)

    # choose some colors
    c.setStrokeColor('0.2,0.5,0.3')
    c.setFillColor('green')
    # draw some lines
    c.line(-50, -50, -50, 3.6*cm)
    c.line(-50, -50, -50+(3.2*cm), -50)
    # draw a rectangle
    c.rect(-50+(0.5*cm), -50+(0.5*cm), -50+(4.2*cm), -50+(5.2*cm), fill=1)
    # Lower corner icon
    # make text go straight up
    c.rotate(90)
    # change color
    c.setFillColor('yellow')
    # say hello (note after rotate the y coord needs to be negative!)
    # define a large font
    c.setFont("Helvetica", 14)
    c.drawString(-50+(0.7*cm), 50+(-2*cm), "TreeGIS")
    c.drawString(-50+(0.7*cm), 50+(-2.5*cm), "Consultants")
    c.showPage()

    for tree in tree_data.iterrows():
        tree_id = tree[0]

        # move the origin up and to the left
        c.translate(inch, inch)

        # choose some colors
        c.setStrokeColor('0.2,0.5,0.3')
        c.setFillColor('green')
        # draw some lines
        c.line(-50, -50, -50, 3.6*cm)
        c.line(-50, -50, -50+(3.2*cm), -50)
        # draw a rectangle
        c.rect(-50+(0.5*cm), -50+(0.5*cm), -50+(4.2*cm), -50+(5.2*cm), fill=1)

        # Align images
        c.drawImage('./output/lidar/'+str(tree_id) +
                    '_aerial.jpg', 200, 450, 10*cm, 10*cm)
        c.drawImage('./output/lidar/'+str(tree_id) +
                    '_plot0deg.jpg', -100, 540, 10*cm, 10*cm)
        c.drawImage('./output/lidar/'+str(tree_id) +
                    '_plot45deg.jpg', -100, 300, 10*cm, 10*cm)
        c.drawImage('./output/lidar/'+str(tree_id) +
                    '_plot90deg.jpg', -100, 60, 10*cm, 10*cm)
        c.drawImage('./output/lidar/'+str(tree_id) +
                    '_ndvi_plot.png', 50, -50, 17.4*cm, 5*cm)
        # Text for trees
        c.setFillColor('black')
        c.setFont("Helvetica-Bold", 16)
        c.drawString(225, 420, "Royal Botannical Gardens")
        c.setFont("Helvetica", 13)
        c.drawString(175, 390, "Tree ID: "+str(tree_id))
        c.drawString(175, 370, "Name: "+str(tree[1]['ComName']))
        c.drawString(175, 350, "Botanic Name: "+str(tree[1]['BotName']))
        c.drawString(175, 330, "Family: "+str(tree[1]['Family']))
        c.drawString(175, 310, "Planted On: "+str(tree[1]['PlantDate']))
        c.drawString(175, 290, "Height: "+str(tree[1]['height']))
        c.drawString(175, 270, "Volume: "+str(tree[1]['tree_volume']))
        c.drawString(175, 250, "Latitude: "+str(tree[1]['lat']))
        c.drawString(175, 230, "Longitude: "+str(tree[1]['long']))
        c.drawString(175, 210, "Garden Bed Name: "+str(tree[1]['RBG_Bed']))

        # Lower corner icon
        # make text go straight up
        c.rotate(90)
        # change color
        c.setFillColor('yellow')

        # say hello (note after rotate the y coord needs to be negative!)
        # define a large font
        c.setFont("Helvetica", 14)
        c.drawString(-50+(0.7*cm), 50+(-2*cm), "TreeGIS")
        c.drawString(-50+(0.7*cm), 50+(-2.5*cm), "Consultants")
        c.showPage()
    c.save()


def main():
    print("------- LiDAR tree analysis -------")
    print("Royal Botanic Gardens, Melbourne, Australia")
    print("RBG boundary file:\t", boundary_shapefile)
    print("Tree/LiDAR info file:\t", las_shapefile)
    print("Output folder location:\t", output_folder)
    crop_the_planet()

    ''' Multitask this to use all the processes available. This works
        fine with this dataset with 12 threads and 32gb RAM. But may
        overload the ram as there seems to be a problem clearing the
        ram after each thread has been run.
    '''
    print("\n------- Scanning LiDAR Data -------")
    run_lidar_multithread()

    ''' Import the CSV data for each LiDAR, created during the previous
        step. If it had been processed already, it is
        skipped, so this file should contain data from a previous process
    '''
    lidar_data = import_lidar_csvs(output_folder)

    # Calculate the NDVI index for each satellite image, on each lidar
    for tif in sorted(glob.iglob(os.path.join(os.getcwd(),
                                              'output', 'ndvi', '*.tif'))):
        ndvi_date = tif.split('/')[-1].split('.')[0]
        lidar_data[ndvi_date] = lidar_data.apply(
            lambda row: ndvi_at_coordinate(
                './output/ndvi/'+ndvi_date+'.ndvi.tif',
                row['utm_x'], row['utm_y']), axis=1)

    # Create NDVI time plots for each lidar and output them
    plot_lidar_data(lidar_data, output_folder)

    # Output everything into a PDF!
    output_pdf(lidar_data, 'RBG_Consultancy.pdf')


if __name__ == "__main__":
    main()
    print("Done!")
