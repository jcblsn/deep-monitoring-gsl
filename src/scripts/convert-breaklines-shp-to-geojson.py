import geopandas as gpd 
import folium
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, polygonize

breaklines = gpd.read_file('../../reference/UGRC-2016-LidAR-project-files/GreatSaltLake_Breaklines.shp')

def convert_3d_to_2d(linestring):
    return LineString([xy[0:2] for xy in list(linestring.coords)])

# get crs
breaklines.crs

breaklines['geometry'] = breaklines['geometry'].apply(convert_3d_to_2d)
gdf_dissolved = breaklines.dissolve(by='B_LINE_TY')
polygons = gpd.GeoSeries()
for geometry in gdf_dissolved.geometry:
    new_polygons = list(polygonize(geometry))
    polygons = polygons.append(gpd.GeoSeries(new_polygons))

poly_gdf = gpd.GeoDataFrame(geometry=polygons)

poly_gdf.plot()
breaklines.plot()

poly_gdf = poly_gdf.set_crs("EPSG:6341")
poly_gdf = poly_gdf.to_crs("EPSG:4326")

m = folium.Map(location=[40.7, -111.8], zoom_start=10, tiles='Stamen Terrain', crs='EPSG3857')
folium.GeoJson(poly_gdf).add_to(m)
folium.GeoJson(breaklines).add_to(m)
m

# save m 
m.save('GreatSaltLake_Breaklines.html')

poly_gdf = poly_gdf.to_crs("EPSG:6341")
poly_gdf.to_file('../../data/definitions/GreatSaltLake_Breaklines_Polygons.geojson', driver='GeoJSON')
