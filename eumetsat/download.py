from dataclasses import dataclass
import os
from multiprocessing import Process, JoinableQueue
import requests
import re
from datetime import datetime, timedelta
import json
from azure.datalake.store import lib, core


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

# Define our start and end dates for temporal subsetting
start_date = datetime(2017, 5, 1, 0, 0, 0)
end_date = datetime(2017, 8, 1, 0, 0, 0)
collection_id = 'EO:EUM:DAT:MSG:MSG15-RSS'
# Number or processes to run with
PROCS = 12

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


class Gen(Process):
    # API base endpoint
    apis_endpoint = "https://api.eumetsat.int/data/search-products/os"

    def __init__(self, task_queue:JoinableQueue, start:datetime, end:datetime):
        Process.__init__(self)
        self.task_queue = task_queue
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

    def run(self):
        while self.start_index < self.total:
            batch_uf = self.get_unfiltered_batch(self.start_index)
            batch = filter(ft, batch_uf)
            # put on queue
            [self.task_queue.put(el) for el in batch]
            self.start_index += self.items_per_page

    def get_unfiltered_batch(self, start_index:int=0):
        response = requests.get(self.apis_endpoint, self.dataset_parameters)
        found_data_sets = response.json()
        return found_data_sets["features"]


class Downloader(Process):
    def __init__(self, task_queue: JoinableQueue, t:EumetsatToken):
        Process.__init__(self)
        self.task_queue = task_queue
        self.t = t
        # Auth against the datalake
        with open("./adls.key") as f:
           adls_info = json.load(f)

        self.adls = core.AzureDLFileSystem(lib.auth(**adls_info), store_name=adls_info["store_name"])

    def run(self):

        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print('Tasks Complete')
                self.task_queue.task_done()
                break
            date = parse_date(next_task["properties"]["date"].split("/")[0])
            dl_url = next_task['properties']['links']['data'][0]['href']
            folder = os.path.join(f"/raw/imagery/eumetsat", collection_id.replace(":", "_"), date.strftime("year=%Y/month=%m/day=%d"))
            self.download(dl_url, folder)
            self.task_queue.task_done()
        return

    def download(self, url, base):
        res = requests.get(url, {"access_token": self.t.token}, stream=True)
        filename = re.findall("\"(.*?)\"", res.headers['Content-Disposition'])[0]
        path = os.path.join(base, filename)
        print(f"{url} -> {path}")
        self.adls.mkdir(f"/{base}")
        with self.adls.open(path, 'wb') as f:
            for c in res.iter_content(chunk_size=1024 * 8):
                f.write(c)


# Queue for interprocess communication
queue = JoinableQueue(PROCS + int(PROCS * 0.20))

# Start Downloader processes
[Downloader(queue, EumetsatToken()).start() for _ in range(PROCS)]
# Start and join generator
g = Gen(queue, start_date, end_date)
g.start()
g.join()

# Join queue and put `final` message to end downloader
[queue.put(None) for _ in range(PROCS)]
queue.join()
print("Done")
