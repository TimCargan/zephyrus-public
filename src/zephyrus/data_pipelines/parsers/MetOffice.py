import functools
from typing import List, Dict
from urllib.parse import quote

import keras_tuner as kt

from src.zephyrus.data_pipelines.parsers.Parser import Parser


class MetOffice(Parser):
    """
    This is the decoder for the new data format with met office images
    ATM This is very slow to read direct from disk, I Think this is an issue with The filesizes and the fact that decode
    has a lot of image movments, the recomdned way is to pre-process the data first

    Data layout
    images are 500x500 pngs [Rainfall, SattilteVis, SatteliteIR]
    A row(example) consists of each UK image layer and a column for each plants metrics in the from {plant}_{metric},
     e.g "kirton_irrad".

    The processing takes the full UK images, slices and resizes and needed then joins it to the plant metrics
    The idea being reading on example can produce ~n examples

    info about the images:
    https://www.metoffice.gov.uk/services/data/datapoint/inspire-layers-detailed-documentation
    """

    """ Constants baked into the data"""
    IM_LAYERS = ["SatelliteVis"] #["Rainfall", "SatelliteVis", "SatelliteIR"]
    SEQ_LENGTH = 120
    IM_SIZE = (500, 500)
    IM_SQ_LENGTH = 41
    IM_SQ_LENGTH_RAINFALL = 121 * 4  # 121h @ 15 min frequency

    @property
    def runbook(self) -> List[Dict]:
        return [
         {"op": "map", "fn": self.parse},
         {"op": "filter", "fn": self.validate},
         {"op": "map", "fn": self.parse_imges},
         {"op": "interleave", "fn": self.melt},
         {"op": "map", "fn": self.shape},
        ]

    def __init__(self, hp: kt.HyperParameters, train=True):
        self.hp = hp
        self.train = train

        self.OUTPUT_IM_SIZE = (500, 500)
        self.OUTPUT_IM_SQ_LENGTH = 16
        self.OUTPUT_IM_SLICE_SIZE = (hp.get("IM_HIGHT"), hp.get("IM_WIDTH"))

        self.out_steps = 24 #self.hp.get("LEAD_STEPS")
        self.out_steps_offset = 12

        int_features = ["time"]
        float_features = ["irrad", "azimuth", "zenith", "elevation", "equation_of_time"]

        features_int = {k: tf.io.FixedLenFeature([self.SEQ_LENGTH], tf.int64) for k in int_features}
        features_float = {k: tf.io.FixedLenFeature([self.SEQ_LENGTH], tf.float32) for k in float_features}
        features = {**features_int, **features_float}
        self._plant_features = features.keys()


        self.plants = [{"lon":-0.70429,"lat":53.32658,"_name":"far dane\'s"},
                       {"lon":-0.93578,"lat":52.7833, "_name":"asfordby b"},]
                     #   {"lon":-0.93765,"lat":52.77801,"_name":"asfordby a"},
                     #   {"lon":-0.09311,"lat":52.93244,"_name":"kirton"},
                     #   {"lon":-1.36974,"lat":52.66413,"_name":"nailstone"}, # Bad Data ?
                     #   {"lon":-4.75253,"lat":50.54756,"_name":"kelly green"},
                     #   {"lon":-4.35646,"lat":50.40197,"_name":"bake solar farm"},
                     #   {"lon":-4.03995,"lat":50.40226,"_name":"newnham"},
                     #   {"lon":-3.71446,"lat":51.53714,"_name":"caegarw"},
                     #   {"lon":-3.47093,"lat":51.39694,"_name":"rosedew"},
                     #   {"lon":0.20657 ,"lat":51.45644,"_name":"moss electrical"},
                     #   {"lon":-0.21804,"lat":52.20184,"_name":"caldecote"},
                     #   {"lon":-0.48715,"lat":52.16442,"_name":"clapham"},
                     #   {"lon":-1.61676,"lat":51.20106,"_name":"lains farm"},
                     #   {"lon":-1.58249,"lat":52.57062,"_name":"magazine"},
                     #   {"lon":-4.74642,"lat":51.67116,"_name":"roberts wall solar farm"},
                     #   {"lon":-3.15562,"lat":51.67448,"_name":"crumlin"},
                     #   {"lon":-2.86451,"lat":52.81214,"_name":"moor"},
                     #   {"lon":-2.36196,"lat":51.71857,"_name":"box road"},
                     #   {"lon":0.05652,"lat":52.75065, "_name":"grange farm"},
                     #   {"lon":-1.49286,"lat":52.75457,"_name":"ashby"},
                     #   {"lon":-1.8042,"lat":52.90477, "_name":"somersal solar farm"},
                     # # {"lon":-2.40676,"lat":51.23981,"_name":"soho farm"}, # No Data
                     #   {"lon":-2.60187,"lat":52.98829,"_name":"combermere farm"}]

        self.plants = [{**v, **{"name": quote(v["_name"]), "xy": latlongtoimage(v["lat"],v["lon"], img=self.OUTPUT_IM_SIZE)}, "shard":i} for i,v in enumerate(self.plants)]
        # self.plants = (self.plants[:10] + self.plants[15:]) if self.train else self.plants[10:15] # Lazy Hack

        features = {f"{p['name']}_{k}": v for p in self.plants for k, v in features.items()}

        layer_type = tf.io.VarLenFeature(tf.string)
        images = {f"{l}": layer_type for l in self.IM_LAYERS}

        date = {f"tz":  tf.io.FixedLenFeature([1], tf.int64)}

        feature = {**date, **features, **images}
        self.feature = feature

        return

    def parse(self, serialized_example):
        """ Read serialized examples and return a dict of tensors"""
        ex = tf.io.parse_single_example(serialized_example, self.feature)
        return ex

    def parse_imges(self, example):
        """ Decode the image layers """
        IM_CHANNEL = self.hp.get("IM_CHANNEL")
        im_dict = {}
        for layer in self.IM_LAYERS:
            imgs = tf.sparse.to_dense(example[f"{layer}"])
            imgs = imgs if layer != "Rainfall" else imgs[::4 * 3]  # Make rainfall 1 every 3 hours
            imgs = imgs[:self.OUTPUT_IM_SQ_LENGTH]
            imgs = imgs[4:12]
            imgs = tf.map_fn(lambda i: tf.io.decode_png(i, IM_CHANNEL), imgs, fn_output_signature=tf.uint8)
            imgs = tf.reshape(imgs, (8, *self.IM_SIZE, IM_CHANNEL)) #self.IM_SQ_LENGTH
            imgs = tf.image.resize(imgs, self.OUTPUT_IM_SIZE)
            imgs = tf.image.convert_image_dtype(imgs, tf.float32)

            im_dict[layer] = imgs

        # We cant modify valuse in the dict directly
        ex = {k:v for k, v in example.items() if k not in self.IM_LAYERS}
        ex = {**im_dict, **ex}
        return ex

    def validate(self, example):
        """
        Validate that the parsed example can be parsed correctly
        Make sure that there are images
        :param example:
        :return:
        """
        v = True
        for layer_name in self.IM_LAYERS:
            c = example[f"{layer_name}"].dense_shape[0]
            target = 41 if layer_name != "Rainfall" else 121 * 4
            if c < target:
                v = False

        return v

    def _center_images(self, xy, images):
        x, y = xy
        target_h, target_w = self.OUTPUT_IM_SLICE_SIZE
        IM_HIGHT, IM_WIDTH = self.OUTPUT_IM_SIZE
        pad_h = IM_HIGHT // 2
        pad_w = IM_WIDTH // 2
        # Calculate the x,y offset for the slice centered on x,y
        n_y = y + pad_h - (target_h //2)
        n_x = x + pad_w - (target_w //2)

        img_slice = {}
        for l, imgs in images.items():
            imgs = imgs[:self.OUTPUT_IM_SQ_LENGTH]
            # Pad the image so a 10x10 -> 20 x 20 with padding of 5 on each edge
            imgs = tf.image.pad_to_bounding_box(imgs, pad_h, pad_w, 2*IM_HIGHT, 2*IM_WIDTH)
            img_slice[l] = tf.image.crop_to_bounding_box(imgs, n_y, n_x, target_h, target_w)

        return img_slice

    def melt(self, example):
        """ Melt an example into a dataset with one entry per plant """
        res = []
        for c, _plant in enumerate(self.plants):
            plant = _plant["name"]
            ex = {f: example[f"{plant}_{f}"] for f in self._plant_features}
            imgs = self._center_images(_plant["xy"], {i: example[i] for i in self.IM_LAYERS})
            meta = {"tz": example["tz"]}

            ex = {**meta, **_plant, **imgs, **ex}
            # TODO: If I vectorized the dict I think can remove the concat reduce step
            # for k, v in ex.items():
            #     if k not in res:
            #         res[k] = []
            #     res[k].append(v)
            # return tf.data.Dataset.from_tensor_slices(res)
            ex = tf.data.Dataset.from_tensors(ex)
            res.append(ex)
        return functools.reduce(lambda a,b: a.concatenate(b), res) # Smush into one dataset



    def scale(self, x,y):
        """
        Add a loss weight vector to the tuple fed to the model, it uses the first dim of y to determine how many steps add
        """
        scale = tf.math.log1p(tf.range(y.shape[0],0, -1, dtype=tf.float32))
        scale = tf.expand_dims(scale, 1)
        return x, y, scale

    def shape(self, example):
        """ Shape a melted example for traning
        Return a tuple with (x:dict, y:tensor)
        """
        # lat = int((example['lat'] - 50) / 64)  # maigc number
        # lon = int((example['lon'] + 5) / 64)
        lat, lon = example["xy"][0], example["xy"][1]

        out_step = 8
        out_slice = slice(self.out_steps_offset, self.out_steps + self.out_steps_offset, 3)
        # Sclar inputs
        input_dict = {"plant": example["name"],
                      "tz": tf.repeat(example["time"][self.out_steps_offset], out_step),
                      "step": tf.range(out_step),
                      "y": example["irrad"][out_slice],
                      "past_irrad": tf.repeat([example["irrad"][:self.out_steps_offset]], out_step, axis=0),
                      "shard": example["shard"],
                      "lat": [lat] * out_step,
                      "lon": [lon] * out_step}

        int_features = ["time"]
        float_features = ["azimuth", "zenith", "elevation", "equation_of_time"]

        for input in float_features:
            input_dict[input] = tf.squeeze(example[f"{input}"])[out_slice]

        et = example["time"][out_slice]
        input_dict["hour_sin"], input_dict["hour_cos"] = sin_cos_scale(et, DAY_SECONDS)
        input_dict["year_sin"], input_dict["year_cos"] = sin_cos_scale(et, YEAR_SECONDS)

        # # Center images over plaths
        image_dict = {}

        for l in self.IM_LAYERS:
            input_dict[l] = example[l]

        return input_dict, input_dict["y"]

