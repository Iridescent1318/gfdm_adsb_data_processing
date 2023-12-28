import pandas as pd
from GfdmInfoJsonReader import GfdmInfoJsonReader

GFDM_INFO_JSON_NAME = 'gt_m_cat_test.json'


def get_gfdm_df(gfdm: GfdmInfoJsonReader) -> pd.DataFrame:
    gfdm_df = gfdm.info_df
    gfdm_df = gfdm_df.drop(columns=['jobtaskid', 'satelliteid'])
    return gfdm_df


def get_cleaned_df(gfdm_df: pd.DataFrame, np_df: pd.DataFrame) -> pd.DataFrame:
    new_df = gfdm_df.merge(np_df, how='right', on='group_name')
    new_df = new_df[new_df.apply(lambda x: abs(x['scenestarttime'] - x['timestamp']) <= GfdmInfoJsonReader.TIME_DIFF and abs(
        x['sceneendtime'] - x['timestamp']) <= GfdmInfoJsonReader.TIME_DIFF, axis=1)]
    new_df = new_df[new_df.apply(lambda x: GfdmInfoJsonReader.check_if_inside_the_polygon(
        x['spatialdata'], (x['longitude'], x['latitude'])), axis=1)]
    new_df['diff'] = new_df.apply(
        lambda x: x['timestamp'] - x['scenestarttime'], axis=1)
    new_df['rel_pos'] = new_df.apply(lambda x: GfdmInfoJsonReader.get_bbox_rel_pos(
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


if __name__ == '__main__':
    gfdm = GfdmInfoJsonReader(GFDM_INFO_JSON_NAME)

    GfdmInfoJsonReader.opensky_query(gfdm.get_groupby_df(), resume=False)

    gfdm_df = get_gfdm_df(gfdm)
    print(gfdm_df)
    np_df = GfdmInfoJsonReader.process_queried_csv('result')
    print(np_df)
    new_df = get_cleaned_df(gfdm_df, np_df)
    print(new_df)

    new_df.to_csv('final.csv')
    final_df = get_groupby_df(new_df)
    final_df.to_csv('final_sceneid.csv')
