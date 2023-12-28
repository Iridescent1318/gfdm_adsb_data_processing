'''
This file processes JSON format ADS-B data downloaded from adsbexchange.
Deprecated now.
'''

import json


class AdsbJsonReader():
    def __init__(self, json_file_path="") -> None:
        try:
            with open(json_file_path, 'r') as f:
                content = json.load(f)
                self.now = content["now"]
                self.messages = content["messages"]
                self.aircraft = content["aircraft"]
        except Exception as e:
            print(e)

    def get_aircraft_key(self, keys):
        ret = []
        for ac in self.aircraft:
            item = tuple([ac[key] if key in ac else None for key in keys])
            ret.append(item)
        return ret


if __name__ == '__main__':
    AIRPORT_POS = {
        "hongkong": (22.308046, 113.918480),
        "tokyo_haneda": (35.553333, 139.781113),
        "tokyo_narita": (35.765786, 140.386347),
        "shanghai_hongqiao": (31.197736, 121.334566),
        "shanghai_pudong": (31.143333, 121.805275),
        "singapore_changi": (1.359167, 103.989441),
        "beijing_capital": (40.072498, 116.597504),
    }
    RANGE = 0.2
    JSON_DIR = "040000Z.json"
    adsb_json_reader = AdsbJsonReader(JSON_DIR)
    key_list = ["hex", "alt_baro", "alt_geom", "gs", "lat", "lon"]
    aircraft_list = adsb_json_reader.get_aircraft_key(key_list)
    for ap, pos in AIRPORT_POS.items():
        lat_min, lat_max = 90, 0
        lon_min, lon_max = 180, -180
        for al in aircraft_list:
            if al[4] is not None and al[5] is not None:
                if -RANGE <= (al[4] - pos[0]) <= RANGE and -RANGE <= (al[5] - pos[1]) <= RANGE:
                    # print(al)
                    lat_min = min(lat_min, al[4])
                    lat_max = max(lat_max, al[4])
                    lon_min = min(lon_min, al[5])
                    lon_max = max(lon_max, al[5])
        print(f"""{lon_min - 0.01:.5f},{lat_max:.5f},{lon_max - 0.01:.5f},{lat_max:.5f},{lon_max - 0.01:.5f},{lat_min:.5f},{lon_min - 0.01:.5f},{lat_min:.5f},{lon_min - 0.01:.5f},{lat_max:.5f}""")
