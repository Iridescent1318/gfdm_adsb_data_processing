'''
The first version of opensky querier.
This class is deprecated because the API is not so convenient.
'''

import time
from typing import Tuple, Any

from tqdm import tqdm
from opensky_api import OpenSkyApi

USERNAME = 'Iridescent1318'
PASSWORD = 'H1guch1Mad0ka'
GFDM_INFO_JSON_NAME = 'gt_m_cat.json'
AIRPORT_POS = {
    "hongkong": (22.308046, 113.918480),
    "tokyo_haneda": (35.553333, 139.781113),
    "tokyo_narita": (35.765786, 140.386347),
    "shanghai_hongqiao": (31.197736, 121.334566),
    "shanghai_pudong": (31.143333, 121.805275),
    "singapore_changi": (1.359167, 103.989441),
    "beijing_capital": (40.072498, 116.597504),
}
POINTS = [v for _, v in AIRPORT_POS.items()]


class OpenSkyQuerier(object):
    def __init__(self) -> None:
        self.api = OpenSkyApi(USERNAME, PASSWORD)

    def query(self, query_tuple: Tuple[Any]):
        df = self.gfdm.time_filter(*query_tuple).get_query_df()
        ret = []
        with tqdm(total=len(df)) as pbar:
            for index, row in df.iterrows():
                try:
                    query_result = self.api.get_states(
                        row['scenestarttime'], None, row['bbox'])
                    if query_result is not None:
                        for qr in query_result.states:
                            ret.append(qr)
                    print(f'{row["sceneid"]} result returned')
                except Exception as e:
                    print(f'An exception has occured, skip: {e}')
                    continue
                finally:
                    pbar.update(1)
                    time.sleep(0.5)
        return ret


if __name__ == '__main__':
    api = OpenSkyApi(USERNAME, PASSWORD)
    # s = api.get_states(1684130944, icao24=None, bbox=(22.3, 22.4, 113.9, 114.0))
    # s = api.get_states()
    # print(s.states[0].icao24)

    s = api.get_flights_from_interval(1684130916, 1684130946)
    print(s)
    # querier = OpenSkyQuerier()
    # ret = querier.query(('scenestarttime', 'gt', '2023-05-15T12:00:00'))
    # print(ret)
