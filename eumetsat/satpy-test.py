import pyresample as pr
from satpy import Scene
from pyproj.crs import CRS
# https://www.dariusgoergen.com/2020-05-10-nat2tif/

def make_pngs(path):
    # TODO we should save this metadata somewhere as we output to PNG so it can get lost
    # set up area
    # UK coords in degrees as per WSG84 [llx, lly, urx, ury]
    #   area_extent = (-7.572168, 49.96, 1.681531, 58.635)
    area_extent = (-12., 48., 5., 61.)
    area_id = "UK"
    description = "Geographical Coordinate System clipped on UK"
    proj_dict = {"proj": "longlat", "ellps": "WGS84", "datum": "WGS84"}
    proj_crs = CRS.from_user_input(4326)  # Target Projection EPSG:4326 standard lat lon geograpic
    output_res = (500, 500)  # Target res in pixels
    area_def = pr.geometry.AreaDefinition.from_extent(area_id, proj_crs, output_res, area_extent)

    # read the file
    reader = "seviri_l1b_native"

    try:
        scn = Scene(filenames={reader: [path]})
        scn.load(scn.all_dataset_names())  # Load all the data inc HRV
        res = scn.resample(area_def, resampler="bilinear", cache_dir='/resample_cache/')
        res.save_datasets(writer="simple_image",
                          filename="{start_time:year=%Y/month=%m/day=%d/time=%H_%M}/format={name}/img.png",
                          format="png", base_dir="/dbfs/mnt/raistore/staging/eumetsat/UK")
    except Exception:
        return False
    return True