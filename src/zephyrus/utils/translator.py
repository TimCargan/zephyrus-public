import platform

# host: {pp: hp}
_paths = {
    "GPU_CLUSTER" : {
        "data":  "/db/psytc3/data",
        "results": "/db/psytc3/results"
    },
    "nimloth": {
        "data":  "/media/tim/Data/riastore",
        "results": "/media/tim/Data/results"
    },
    "grond": {
        "data":  "/home/psytc3/data",
        "results": "/home/psytc3/results"
    },
    "atlas": {
        "data": "/mnt/d/data",
        "results": "/mnt/d/results"
    },

    }


def get_path(folder: str) -> str:
    """
    Gets the local path for the project folder
    if none exists an out of bounds error is thrown
    TODO: make this better
    :param folder:
    :return:
    """
    host = platform.node().lower()
    # Cluser check
    if "cs.nott.ac.uk" in host:
        host = "GPU_CLUSTER"
    # path_dict = [x for x in _paths if x["host"].lower() == host and x["project_path"] == folder]
    return _paths[host][folder]
