import json
import datetime
import time
import os
from typing import Tuple, Sequence, TypeVar

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from traffic.data import opensky
from tqdm import tqdm


GFDM_INFO_JSON_NAME = 'gt_m_cat_test.json'
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
TIME_DIFF = 1
EPS = 1e-6

JSON_UTC_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
CSV_UTC_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
T = TypeVar('T', int, float)


class GfdmInfoJsonReader(object):
    '''
    The class of GFDM JSON metadata reader & processor.

    Attributes:
        fname (str): the filename of the JSON file.
    '''
    @staticmethod
    def convert_utcstr_to_timestamp(utc_time: str, format_str: str) -> int:
        '''
        Convert a UTC time string to a UNIX timestamp. 
        The UTC string format is like YYYY-mm-ddTHH:MM:SS, e.g. 2023-01-01T08:00:00.0000
        This format is consistent with that of GFDM metadata json files.

        Args:
            utc_time (str): The formatted UTC time string.

        Returns:
            An integer type of converted UNIX timestamp.
        '''
        utc_time_part = utc_time_part = utc_time[:
                                                 23] if '.' in utc_time else utc_time[:19] + '.000'
        dt = datetime.datetime.strptime(
            utc_time_part, format_str).replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp())

    @staticmethod
    def convert_sds_to_tuple(spatial_data_str: str) -> (Tuple[Tuple[T]], Tuple[T]):
        '''
        Convert a spatial data string containing four longitude-latitude coordinates to a 
        four-element tuple (min_longitude, min_latitude, max_longitude, max_latitude).

        Args:
            spatial_data_str (str): a spatial data string containing four coordinates,
                e.g. ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), where xs and ys are 
                longitudes and latitudes respectively.

        Returns:
            A tuple of two elements. The first element is the tuple object deserialized from 
            the string, and the second is the four-element tuple.
        '''
        pt_tuples = eval(f"tuple({spatial_data_str})")
        max_lon = max([tp[0] for tp in pt_tuples])  # east
        min_lon = min([tp[0] for tp in pt_tuples])  # west
        max_lat = max([tp[1] for tp in pt_tuples])  # north
        min_lat = min([tp[1] for tp in pt_tuples])  # south
        # (west, south, east, north)
        return pt_tuples, (min_lon, min_lat, max_lon, max_lat)

    @staticmethod
    def get_max_bbox_tuple(bbox_list: Sequence[Sequence[T]]) -> Tuple[T]:
        '''
        Given a list of an oblique bounding box coordinates, return its minimal circumscribed rectangle 
        whose edges are parallel to lines of longitude and latitude.

        Note: 
            This method is not applicable to boxes across the 180th meridian (i.e. the 180° line).

        Args:
            bbox_list: a list containing four longitude-latitude coordinates.

        Returns:
            A tuple like (min_longitude, min_latitude, max_longitude, max_latitude), or equivalently 
            (west, south, east, north).
        '''
        ret = [180, 90, -180, -90]
        for bl in bbox_list:
            ret[0] = min(ret[0], bl[0])  # west
            ret[1] = min(ret[1], bl[1])  # south
            ret[2] = max(ret[2], bl[2])  # east
            ret[3] = max(ret[3], bl[3])  # north
        return tuple(ret)

    @staticmethod
    def check_if_inside_the_polygon(polygon: Sequence[Sequence[T]], point: Sequence[T]) -> bool:
        '''
        Check if a given coordinate is inside a given polygon.

        Args:
            polygon: a list containing points of a polygon in (anti-)clockwise order.
            point: the coordinate to be checked.

        Returns:
            True if the coordinate is inside the polygon, otherwise False.
        '''
        assert (len(polygon) > 0)
        assert (len(polygon[0]) == 2)
        assert (len(point) == 2)
        point_shape = Point(point[0], point[1])
        polygon_shape = Polygon(polygon)
        return polygon_shape.contains(point_shape)

    @staticmethod
    def get_bbox_rel_pos(bbox: Sequence[Sequence[T]], lon: T, lat: T) -> Tuple[T]:
        '''
        Get the relative position of a given lon-lat coordinate in a bounding box

        Args:
            bbox: a list containing four longitude-latitude coordinates
            lon: the longitude of the point
            lat: the latitude of the point

        Returns:
            A tuple with two elements. The first element is the relative position with respect to longitude,
            and the second latitude.
        '''
        return ((lon - bbox[0]) / (bbox[2] - bbox[0] + EPS), (lat - bbox[1]) / (bbox[3] - bbox[1] + EPS))

    time_attributes = {'scenestarttime', 'sceneendtime'}
    time_filtering_rels = {'lt', 'le', 'equal', 'ge', 'gt'}

    @staticmethod
    def json_utc_converter(
        s): return GfdmInfoJsonReader.convert_utcstr_to_timestamp(s, JSON_UTC_FORMAT)

    @staticmethod
    def csv_utc_converter(
        s): return GfdmInfoJsonReader.convert_utcstr_to_timestamp(s, CSV_UTC_FORMAT)

    def __init__(self, fname):
        self.fname = fname
        with open(self.fname) as f:
            self.info = json.load(f)['RECORDS']
        for record in self.info:
            record['group_name'] = record['sceneid'][:33]
            # record['scenestarttime_dt'] = record['scenestarttime']
            # record['sceneendtime_dt'] = record['sceneendtime']
            record['scenestarttime'] = GfdmInfoJsonReader.json_utc_converter(
                record['scenestarttime'])
            record['sceneendtime'] = GfdmInfoJsonReader.json_utc_converter(
                record['sceneendtime'])
            record['spatialdata'], record['bbox'] = GfdmInfoJsonReader.convert_sds_to_tuple(
                record['spatialdata'])

        self.info_df = pd.DataFrame.from_dict(self.info)
        self.query_df = pd.DataFrame.copy(self.info_df)

        self.groupby_df = self.info_df.groupby('group_name').agg({
            'scenestarttime': np.min,
            'sceneendtime': np.max,
            'bbox': GfdmInfoJsonReader.get_max_bbox_tuple,
            'spatialdata': lambda x: x,
            'sceneid': lambda x: x
        })
        self.groupby_df['starttimelist'] = self.info_df.groupby('group_name')[
            'scenestarttime']
        self.groupby_df['endtimelist'] = self.info_df.groupby('group_name')[
            'sceneendtime']

        self.filter_list = []

    def reset_query(self) -> None:
        '''
        Reset the query dataframe to initial data records.
        '''
        self.query_df = pd.DataFrame.copy(self.info_df)

    def get_query_df(self) -> pd.DataFrame:
        return self.query_df

    def get_groupby_df(self) -> pd.DataFrame:
        return self.groupby_df

    def time_filter(self, time_attribute: str, rel: str, utc_time_str: str) -> "GfdmInfoJsonReader":
        '''
        Filter records which satisfy the given time condition.
        '''
        df = self.query_df
        assert (time_attribute in GfdmInfoJsonReader.time_attributes)
        assert (rel in GfdmInfoJsonReader.time_filtering_rels)
        utc_timestamp = GfdmInfoJsonReader.json_utc_converter(utc_time_str)
        criteria = None
        if rel == 'gt':
            criteria = df[time_attribute] > utc_timestamp
        elif rel == 'ge':
            criteria = df[time_attribute] >= utc_timestamp
        elif rel == 'equal':
            criteria = df[time_attribute] == utc_timestamp
        elif rel == 'le':
            criteria = df[time_attribute] <= utc_timestamp
        elif rel == 'lt':
            criteria = df[time_attribute] < utc_timestamp
        self.query_df = df[criteria]
        return self

    def space_filter(self, points) -> "GfdmInfoJsonReader":
        '''
        Filter records which contain at least one point from the input points.
        '''
        df = self.query_df
        criteria = df['spatialdata'].apply(lambda x: True if len(
            points) == 0 else np.any([GfdmInfoJsonReader.check_if_inside_the_polygon(x, point) for point in points]))
        self.query_df = df[criteria]
        return self

    @staticmethod
    def opensky_query(df: pd.DataFrame) -> None:
        '''

        '''
        with open('done.txt', 'r', encoding='utf-8') as f_in:
            done = set(l[:-1] for l in f_in.readlines())
        try:
            with tqdm(total=df.shape[0] - len(done)) as pbar:
                for index, row in df.iterrows():
                    if index in done:
                        continue
                    start_time = row['scenestarttime']
                    stop_time = row['sceneendtime']
                    bounds = row['bbox']
                    traffic = opensky.history(
                        start=start_time,
                        stop=stop_time,
                        bounds=bounds
                    )
                    if traffic is not None:
                        traffic.data.to_csv(f'result\\{index}.csv')
                        print(f"{index} traffic information found and saved")
                    with open('done.txt', 'a+', encoding='utf-8') as f_in:
                        f_in.write(index + '\n')
                    time.sleep(0.5)
                    pbar.update(1)
        except Exception as e:
            print(e)

    @staticmethod
    def process(path: str) -> pd.DataFrame:
        file_list = os.listdir(path)
        np_list = []
        data_head = None
        for fl in file_list:
            fname = fl[:33]
            file_df = pd.read_csv(os.path.join(path, fl))
            file_df['group_name'] = fname
            if data_head is None:
                data_head = file_df.columns.values
            file_list = file_df.to_numpy()
            np_list.append(file_list)
        np_stack_list = np.vstack(np_list)

        ret = pd.DataFrame(np_stack_list, columns=data_head)
        ret = ret.drop(columns=['Unnamed: 0', 'alert', 'spi', 'squawk'])
        ret['last_position'] = ret['last_position'].apply(
            GfdmInfoJsonReader.csv_utc_converter)
        ret['hour'] = ret['hour'].apply(
            GfdmInfoJsonReader.csv_utc_converter)
        ret['timestamp'] = ret['timestamp'].apply(
            GfdmInfoJsonReader.csv_utc_converter)
        ret = ret[ret['groundspeed'].notnull()]
        ret = ret[ret.apply(lambda x: abs(x['last_position'] -
                            x['timestamp']) <= TIME_DIFF, axis=1)]

        return ret


if __name__ == '__main__':
    gfdm = GfdmInfoJsonReader(GFDM_INFO_JSON_NAME)

    GfdmInfoJsonReader.opensky_query(gfdm.get_groupby_df())

    gfdm_df = gfdm.info_df
    gfdm_df = gfdm_df.drop(columns=['jobtaskid', 'satelliteid'])
    print(gfdm_df)
    np_df = GfdmInfoJsonReader.process('result')
    print(np_df)

    new_df = gfdm_df.merge(np_df, how='right', on='group_name')
    new_df = new_df[new_df.apply(lambda x: abs(x['scenestarttime'] - x['timestamp'])
                                 <= TIME_DIFF and abs(x['sceneendtime'] - x['timestamp']) <= TIME_DIFF, axis=1)]
    new_df = new_df[new_df.apply(lambda x: GfdmInfoJsonReader.check_if_inside_the_polygon(
        x['spatialdata'], (x['longitude'], x['latitude'])), axis=1)]
    new_df['diff'] = new_df.apply(
        lambda x: x['timestamp'] - x['scenestarttime'], axis=1)
    new_df['rel_pos'] = new_df.apply(lambda x: GfdmInfoJsonReader.get_bbox_rel_pos(
        x['bbox'], x['longitude'], x['latitude']), axis=1)
    print(new_df)
    new_df.to_csv('final.csv')

    ndf_groupby = new_df.groupby('sceneid')
    ndf = ndf_groupby.agg({
        'scenestarttime': 'min',
        'sceneendtime': 'max',
        # 'bbox': lambda x: x[0],
        # 'spatialdata': lambda x: x,
        # 'sceneid': lambda x: x
        'rel_pos': lambda x: x
    })
    ndf['plane_count'] = ndf_groupby.agg({
        'icao24': lambda x: len(set(x)),
    })
    ndf.sort_values(by='plane_count', ascending=False).to_csv(
        'final_sceneid.csv')
