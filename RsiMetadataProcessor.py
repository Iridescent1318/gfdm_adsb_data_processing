import json
import datetime
import time
import os
import logging
import shelve
import hashlib
from typing import Tuple, Sequence, TypeVar, Any

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from traffic.data import opensky
from tqdm import tqdm

T = TypeVar('T')
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RsiMetadataProcessor(object):
    '''
    The class of GFDM JSON metadata reader & processor.

    Attributes:
        fpath (str): the filepath of the JSON file.
    '''
    TARGET_SCENE_ID_PATH = 'rmp_cache'
    TIME_ATTRIBUTES = {'scenestarttime', 'sceneendtime'}
    TIME_FILTER_RELS = {'lt', 'le', 'equal', 'ge', 'gt'}
    TIME_DIFF = 1
    EPS = 1e-6
    JSON_UTC_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
    CSV_UTC_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

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
        utc_time_part = utc_time[:23] if '.' in utc_time else utc_time[:19] + '.000'
        dt = datetime.datetime.strptime(
            utc_time_part, format_str).replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp())

    @staticmethod
    def convert_sds_to_tuple(spatial_data_str: str) -> Tuple[Tuple[Tuple[T]], Tuple[T]]:
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
        return ((lon - bbox[0]) / (bbox[2] - bbox[0] + RsiMetadataProcessor.EPS),
                (lat - bbox[1]) / (bbox[3] - bbox[1] + RsiMetadataProcessor.EPS))

    @staticmethod
    def json_utc_converter(s: int | float | str | None) -> int:
        if s is None:
            return -1
        elif isinstance(s, float):
            return int(s)
        elif isinstance(s, int):
            return s
        return RsiMetadataProcessor.convert_utcstr_to_timestamp(s, RsiMetadataProcessor.JSON_UTC_FORMAT)

    @staticmethod
    def csv_utc_converter(s: int | float | str | None) -> int:
        if s is None:
            return -1
        elif isinstance(s, float):
            return int(s)
        elif isinstance(s, int):
            return s
        elif isinstance(s, pd.Timestamp):
            return s.value // 10 ** 9
        return RsiMetadataProcessor.convert_utcstr_to_timestamp(s, RsiMetadataProcessor.CSV_UTC_FORMAT)

    def query_historical_adsb(self, save_output: bool = False, output_dir: str = './target_scene_csv') -> dict:
        '''
        Get historical flight information using cleaned JSON metadata by the API of `traffic`.

        The JSON metadata is grouped by sceneid prefix [0:position_of_last_underscore] (e.g. 
        DM01_PMS_013469_20230102_KS5M1_02_022 -> DM01_PMS_013469_20230102_KS5M1_02) and cleaned
        during the initialization of the class. Each group will produce a CSV file containing
        all ADS-B messages during the imaging period.

        Note: if the data is too much, the inquiry may stuck occasionally. This issue is to be
            troubleshot in the future. Just try more times with resume=True.

        Args:
            output_dir: the output directory path of saved CSV files.

        Returns:
            A dict whose keys are ids of scenes containing at least one plane and values are correlated pandas dataframe.
        '''
        df = self.get_groupby_df()
        try:
            if not os.path.exists(RsiMetadataProcessor.TARGET_SCENE_ID_PATH):
                os.mkdir(RsiMetadataProcessor.TARGET_SCENE_ID_PATH)
            shelve_fname = os.path.join(
                RsiMetadataProcessor.TARGET_SCENE_ID_PATH, self.fname + '_' + self.id_suffix)
            with shelve.open(shelve_fname, writeback=True) as db:
                if 'target_scene_ids' not in db:
                    db['target_scene_ids'] = dict()
                with tqdm(total=df.shape[0] - len(db['target_scene_ids'])) as pbar:
                    for index, row in df.iterrows():
                        if index in db['target_scene_ids']:
                            continue
                        traffic = opensky.history(
                            start=row['scenestarttime'],
                            stop=row['sceneendtime'],
                            bounds=row['bbox']
                        )
                        if traffic is not None:
                            if save_output:
                                if not os.path.exists(output_dir):
                                    logger.warning(
                                        f"The output path {output_dir} does not exist. Trying to mkdir.")
                                    os.mkdir(output_dir)
                                traffic.data.to_csv(os.path.join(
                                    output_dir, f'{index}.csv'))
                            db['target_scene_ids'][index] = traffic.data
                            logger.info(
                                f"{index} traffic information found and saved")
                        time.sleep(0.1)
                        pbar.update(1)
                self.target_scene_ids = dict(db['target_scene_ids'])
                logger.debug(db['target_scene_ids'])
        except Exception as e:
            logger.exception(e)
        return self.target_scene_ids

    @staticmethod
    def process_queried_csv(csv_dir_path: str) -> pd.DataFrame:
        '''
        Process all the CSV files produced by opensky_query().

        Args:
            csv_dir_path: the CSV file directory path.

        Returns:
            A pandas dataframe containing cleaned flight data.
        '''
        file_list = os.listdir(csv_dir_path)
        np_list = []
        data_head = None
        for fl in file_list:
            fname = fl[:33]
            file_df = pd.read_csv(os.path.join(csv_dir_path, fl))
            file_df['group_name'] = fname
            if data_head is None:
                data_head = file_df.columns.values
            file_list = file_df.to_numpy()
            np_list.append(file_list)
        np_stack_list = np.vstack(np_list)

        # Convert to dataframe and cleaning
        ret = pd.DataFrame(np_stack_list, columns=data_head)
        ret = ret.drop(columns=['Unnamed: 0', 'alert', 'spi', 'squawk'])
        # Convert to UNIX timestamp
        ret['last_position'] = ret['last_position'].apply(
            RsiMetadataProcessor.csv_utc_converter)
        ret['hour'] = ret['hour'].apply(
            RsiMetadataProcessor.csv_utc_converter)
        ret['timestamp'] = ret['timestamp'].apply(
            RsiMetadataProcessor.csv_utc_converter)
        # Drop rows with null ground speed
        ret = ret[ret['groundspeed'].notnull()]
        # last_position denotes the last time when the ADS-B message of the plane is recorded
        # This is to filter 'fresh' messages
        ret = ret[ret.apply(lambda x: abs(x['last_position'] -
                            x['timestamp']) <= RsiMetadataProcessor.TIME_DIFF, axis=1)]

        return ret

    def filter_target_scenes(self) -> pd.DataFrame:
        data_head = None
        if len(self.target_scene_ids) > 0:
            target_scene_df_list = []
            for scene_id, df in self.target_scene_ids.items():
                group_name = scene_id[:33]
                df['group_name'] = group_name
                if data_head is None:
                    data_head = df.columns.values
                scene_list = df.to_numpy()
                target_scene_df_list.append(scene_list)

            target_scene_df = pd.DataFrame(
                np.vstack(target_scene_df_list), columns=data_head)
            target_scene_df = target_scene_df.drop(
                columns=['alert', 'spi', 'squawk'])
            # Convert to UNIX timestamp
            target_scene_df['last_position'] = target_scene_df['last_position'].apply(
                RsiMetadataProcessor.csv_utc_converter)
            target_scene_df['hour'] = target_scene_df['hour'].apply(
                RsiMetadataProcessor.csv_utc_converter)
            target_scene_df['timestamp'] = target_scene_df['timestamp'].apply(
                RsiMetadataProcessor.csv_utc_converter)
            # Drop rows with null ground speed
            target_scene_df = target_scene_df[target_scene_df['groundspeed'].notnull(
            )]
            # last_position denotes the last time when the ADS-B message of the plane is recorded
            # This is to filter 'fresh' messages
            fresh_target_scene_df = target_scene_df.apply(lambda x: abs(x['last_position'] -
                                                                        x['timestamp']) <= RsiMetadataProcessor.TIME_DIFF, axis=1)
            target_scene_df = target_scene_df[fresh_target_scene_df]

            return target_scene_df
        else:
            return None

    def __init__(self, fpath) -> None:
        self.fpath = fpath
        self.fname, _ = os.path.splitext(os.path.basename(fpath))
        self.id_suffix = hashlib.md5(
            self.fname.encode("utf-8")).hexdigest()[-4:]
        self.target_scene_ids = dict()
        with open(self.fpath) as f:
            self.info = json.load(f)['RECORDS']
        for record in self.info:
            record['group_name'] = record['sceneid'][:33]
            record['scenestarttime'] = RsiMetadataProcessor.json_utc_converter(
                record['scenestarttime'])
            record['sceneendtime'] = RsiMetadataProcessor.json_utc_converter(
                record['sceneendtime'])
            record['spatialdata'], record['bbox'] = RsiMetadataProcessor.convert_sds_to_tuple(
                record['spatialdata'])

        self.original_df = pd.DataFrame.from_dict(self.info)
        self.query_df = pd.DataFrame.copy(self.original_df)

        self.groupby_df = self.original_df.groupby('group_name').agg({
            'scenestarttime': 'min',
            'sceneendtime': 'max',
            'bbox': RsiMetadataProcessor.get_max_bbox_tuple,
            'spatialdata': lambda x: x,
            'sceneid': lambda x: x
        })
        self.groupby_df['starttimelist'] = self.original_df.groupby('group_name')[
            'scenestarttime']
        self.groupby_df['endtimelist'] = self.original_df.groupby('group_name')[
            'sceneendtime']

        self.filter_list = []

    def reset_query(self) -> None:
        '''
        Reset the query dataframe to initial data records.
        '''
        self.query_df = pd.DataFrame.copy(self.original_df)

    def get_query_df(self) -> pd.DataFrame:
        return self.query_df

    def get_groupby_df(self) -> pd.DataFrame:
        return self.groupby_df

    def get_info_df_dropping_task_id(self) -> pd.DataFrame:
        return pd.DataFrame.copy(self.original_df).drop(columns=['jobtaskid', 'satelliteid'])

    def time_filter(self, time_attribute: str, rel: str, utc_time_str: str) -> "RsiMetadataProcessor":
        '''
        Filter records which satisfy the given time condition.

        Args:
            time_attribute: time related attributes. Takes 'scenestarttime' or 'sceneendtime'.
            rel: the comparator in the criteria. Takes 'gt'(greater than), 'ge'(greater than or equal to),
                'equal', 'le'(less than or equal) and 'lt'(less than).
            utc_time_str: the UTC time string from JSON metadata in '%Y-%m-%dT%H:%M:%S'

        Returns:
            The object itself (to support method chaining).
        '''
        df = self.query_df
        assert (time_attribute in RsiMetadataProcessor.TIME_ATTRIBUTES)
        assert (rel in RsiMetadataProcessor.TIME_FILTER_RELS)
        utc_timestamp = RsiMetadataProcessor.json_utc_converter(utc_time_str)
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

    def space_filter(self, points: Sequence[Sequence[T]]) -> "RsiMetadataProcessor":
        '''
        Filter records which contain at least one point from the input points.

        Args:
            points: a sequence of input points.

        Returns:
            The object itself (to support method chaining).
        '''
        df = self.query_df
        criteria = df['spatialdata'].apply(lambda x: True if len(
            points) == 0 else np.any([RsiMetadataProcessor.check_if_inside_the_polygon(x, point) for point in points]))
        self.query_df = df[criteria]
        return self


def get_gfdm_df(rsi_metadata_processor: RsiMetadataProcessor) -> pd.DataFrame:
    gfdm_df = rsi_metadata_processor.original_df
    gfdm_df = gfdm_df.drop(columns=['jobtaskid', 'satelliteid'])
    return gfdm_df


def get_cleaned_df(gfdm_df: pd.DataFrame, np_df: pd.DataFrame) -> pd.DataFrame:
    new_df = gfdm_df.merge(np_df, how='right', on='group_name')
    new_df = new_df[new_df.apply(lambda x: abs(x['scenestarttime'] - x['timestamp']) <= RsiMetadataProcessor.TIME_DIFF and abs(
        x['sceneendtime'] - x['timestamp']) <= RsiMetadataProcessor.TIME_DIFF, axis=1)]
    new_df = new_df[new_df.apply(lambda x: RsiMetadataProcessor.check_if_inside_the_polygon(
        x['spatialdata'], (x['longitude'], x['latitude'])), axis=1)]
    new_df['diff'] = new_df.apply(
        lambda x: x['timestamp'] - x['scenestarttime'], axis=1)
    new_df['rel_pos'] = new_df.apply(lambda x: RsiMetadataProcessor.get_bbox_rel_pos(
        x['bbox'], x['longitude'], x['latitude']), axis=1)
    return new_df


def get_groupby_df(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    ndf_groupby = cleaned_df.groupby('sceneid')
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
    return ndf.sort_values(by='plane_count', ascending=False)


if __name__ == "__main__":
    GFDM_INFO_JSON_PATH = 'gt_m_cat_test.json'

    rmp = RsiMetadataProcessor(GFDM_INFO_JSON_PATH)
    rmp.query_historical_adsb()

    gfdm_df = get_gfdm_df(rmp)
    logging.debug(gfdm_df)
    np_df = rmp.filter_target_scenes()
    logging.debug(np_df)
    new_df = get_cleaned_df(gfdm_df, np_df)
    logging.debug(new_df)

    new_df.to_csv('final.csv')
    final_df = get_groupby_df(new_df)
    final_df.to_csv('final_sceneid.csv')
