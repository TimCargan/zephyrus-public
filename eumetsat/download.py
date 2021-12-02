from dataclasses import dataclass
import os
from multiprocessing import Process, JoinableQueue
import requests
import re
import argparse

from datetime import datetime, timedelta
import json
from azure.datalake.store import lib, core

from zephyrus.utils.standard_logger import build_logger
from zephyrus.utils import translator as t
from pyproj.crs import CRS
from satpy import Scene
from PIL import Image
import pyresample as pr


@dataclass
class EumetsatToken:
    def __init__(self, key_path:str="./eumetsat.key"):
        with open(key_path, "r") as f:
            self._key = json.load(f)
        self._last_load = datetime(2017, 5, 1, 0, 0, 0)

    @property
    def token(self) -> str:
        pass

    @token.getter
    def token(self) -> str:
        if (self._last_load - datetime.utcnow()) < timedelta(minutes=50):
            self._last_load = datetime.utcnow()
            self._token = self._load_token()
        return self._token

    def _load_token(self) -> str:
        token = requests.post("https://api.eumetsat.int/token", data="grant_type=client_credentials",
                              auth=(self._key["username"], self._key["password"]))
        return token.json()["access_token"]

def parse_date(ds:str) -> datetime:
    try:
        date = datetime.strptime(ds, "%Y-%m-%dT%H:%M:%S.%fZ")
    except:
        date = datetime.strptime(ds, "%Y-%m-%dT%H:%M:%SZ")
    return date

def ft(x):
    ds = x["properties"]["date"].split("/")[0]
    date = parse_date(ds)
    return date.minute == 0


class Extract(Process):
    def __init__(self, files: JoinableQueue):
        Process.__init__(self)
        self.files = files
        self.logger = build_logger("Extractor")

    def run(self):
        while True:
            next_task = self.files.get()
            if next_task is None:
                self.logger.info('Tasks Complete')
                self.files.task_done()
                break

            path = next_task
            self.make_pngs(path)
            self.files.task_done()

    def make_pngs(self, path):
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
            self.logger.info(f"Loading {path}")
            scn = Scene(filenames={reader: [path]})
            scn.load(scn.all_dataset_names())  # Load all the data inc HRV
            res = scn.resample(area_def, resampler="bilinear") #cache_dir='/resample_cache/'
            res.save_datasets(writer="simple_image",
                              filename="{start_time:year=%Y/month=%m/day=%d/time=%H_%M}/format={name}/img.png",
                              format="png", base_dir=EXTRACT_BASE_DIR)

            # Delete Nat file, keep disk space free as they are big
            self.logger.info(f"Removing {path}")
            os.remove(path)
        except Exception as e:
            self.logger.error(e)
            return False
        return True


class Gen(Process):
    # API base endpoint
    apis_endpoint = "https://api.eumetsat.int/data/search-products/os"

    def __init__(self, task_queue:JoinableQueue, collection_id, start:datetime, end:datetime):
        Process.__init__(self)
        self.task_queue = task_queue
        self.logger = build_logger("Gen")
        self.items_per_page = 100
        self.start_index = 0
        self.dataset_parameters = {'format': 'json',
                              'pi': collection_id,
                              "c": self.items_per_page,
                              "si": self.start_index,
                              'dtstart': start.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                              'dtend': end.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                              }
        self.total = requests.get(self.apis_endpoint, self.dataset_parameters).json()['properties']['totalResults']
        self.logger.info(f"Total: {self.total}")

    def run(self):
        try:
            self.loop()
        except Exception as e:
            self.logger.error(e)
        finally:
            self.logger.info(f"Added {self.start_index} / {self.total} els to queue ")

    def loop(self):
        while self.start_index < self.total:
            self.logger.info(f"{self.start_index} / {self.total}")
            batch_uf = self.get_unfiltered_batch(self.start_index)
            batch = filter(ft, batch_uf)
            # put files on queue
            for el in batch:
                ds = el["properties"]["date"].split("/")[0]
                date = parse_date(ds)
                self.logger.info(f"Adding File {date.strftime('year=%Y/month=%m/day=%d/time=%H_%M')} to DL Queue")
                self.task_queue.put(el)
            self.start_index += self.items_per_page

    def get_unfiltered_batch(self, start_index:int=0):
        self.dataset_parameters["si"] = start_index #TOTO this is super gross
        response = requests.get(self.apis_endpoint, self.dataset_parameters)
        # Log Failed responses
        if not response.ok:
            self.logger.error(f"Request for {start_index} Failed: {response.text}")
        found_data_sets = response.json()
        return found_data_sets["features"]


class Downloader(Process):
    def __init__(self, task_queue: JoinableQueue, file_queue: JoinableQueue, t:EumetsatToken):
        Process.__init__(self)
        self.task_queue = task_queue
        self.file_queue = file_queue
        self.t = t
        self.logger = build_logger("Downloader")
        # Auth against the datalake
        # with open("./adls.key") as f:
        #    adls_info = json.load(f)
        #
        # self.adls = core.AzureDLFileSystem(lib.auth(**adls_info), store_name=adls_info["store_name"])

    def run(self):
        while True:
            file_path = None
            next_task = None
            try:
                next_task = self.task_queue.get()
                if next_task is None:
                    self.logger.info('Tasks Complete')
                    self.task_queue.task_done()
                    break
                file_path = self._run(next_task)
            except Exception as e:
                self.logger.error(f"Error on {next_task}")
                self.logger.error(e)
            finally:
                self.task_queue.task_done()
                if file_path:
                    self.file_queue.put(file_path)

    def _run(self, next_task) -> str:
        date = parse_date(next_task["properties"]["date"].split("/")[0])
        sip_ents = next_task['properties']['links']['sip-entries']
        nat_file = filter(lambda x: x['mediaType'] == 'application/octet-stream', sip_ents)
        dl_url = list(nat_file)[0]['href']
        # f"/raw/imagery/eumetsat" adls prefiex
        folder = os.path.join(collection_id.replace(":", "_"), date.strftime("year=%Y/month=%m/day=%d/time=%H_%M"))
        file_path = self.download(dl_url, folder)
        return file_path

    def download(self, url, base) -> str:
        res = requests.get(url, {"access_token": self.t.token}, stream=True)
        filename = re.findall("\"(.*?)\"", res.headers['Content-Disposition'])[0]
        path = os.path.join(base, filename)
        self.logger.info(f"{url} -> {path}")
        # self.adls.mkdir(f"/{base}")
        dir = os.path.join(DL_BASE_DIR, base)
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, filename)
        with open(path, 'wb') as f:
            for c in res.iter_content(chunk_size=1024 * 8):
                f.write(c)
        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model")
    parser.add_argument('--dl', default=1, type=int, help="Number of download procs to run")
    parser.add_argument('--ep', default=2, type=int, help="Number of extractor procs to run")

    args = parser.parse_args()

    DL_PROCS = args.dl
    EX_PROCS = args.ep

    DL_BASE_DIR = os.path.join(t.get_path("data"), "EUMETSAT/RAW")
    EXTRACT_BASE_DIR = os.path.join(t.get_path("data"), "EUMETSAT/UK-EXT")

    logger = build_logger("Main")

    # Define our start and end dates for temporal subsetting
    start_date = datetime(2019, 1, 1, 0, 0, 0)
    end_date = datetime(2019, 3, 8, 0, 0, 0)
    # MSG15-RSS
    collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI'


    # Queue for interprocess communication
    url_q = JoinableQueue(DL_PROCS + int(DL_PROCS * 0.20))
    file_q = JoinableQueue(EX_PROCS + int(EX_PROCS * 0.20))

    #TODO: use a threadpool and aync maps for this, why are we managing it ourselvs?
    logger.info("Starting Processes")
    # Start Downloader processes
    ex = [Extract(file_q) for _ in range(EX_PROCS)]
    [e.start() for e in ex]
    [Downloader(url_q, file_q, EumetsatToken()).start() for _ in range(DL_PROCS)]

    logger.info(f"Starting Gen for {start_date} to {end_date}")
    # Start and join generator
    g = Gen(url_q, collection_id, start_date, end_date)
    g.start()
    g.join() # wait for generator to add all urls
    # Join queue and put `final` message to end downloader

    logger.info(f"Added Termination Singles to DOWNLOAD queue")
    [url_q.put(None) for _ in range(DL_PROCS)]
    url_q.join() # wait for files to download
    logger.info("URL Queue Done")

    logger.info(f"Added Termination Singles to Extract queue")
    [file_q.put(None) for _ in range(EX_PROCS)]
    [e.join() for e in ex]
    logger.info("All Procs Done")
#dding File year=2018/month=02/day=23/time=20_00