'''
This py file is for speed/azimuth calculation of moving objects (planes).
'''

import numpy as np
import pandas as pd

FNAME = 'gfdm_pos.csv'
RES = 2
TIMES = [0.4848, 0.5220, 0.5519, 0.5817, 0, 0.0373, 0.0671, 0.1044]
SPEED_GTS = [509, 411, 455, 430, 436, 414, 172, 450]
AZIMUTH_GTS = [64.24, 234.80, 250.49, 232.94, 129.60, 310, 321.13, 351.18]


def calc_xyspeed(line):
    xyspeed = np.zeros((4, 2))
    for i in range(4):
        xyspeed[i][0] = (line[i] - line[7]) * RES / (TIMES[i] - TIMES[7])
        xyspeed[i][1] = (line[i + 8] - line[15]) * RES / (TIMES[i] - TIMES[7])
    xyspeed, std = np.mean(xyspeed, axis=0), np.std(xyspeed, axis=0)
    return xyspeed, std


def calc_speed_azimuth(xy):
    speed = np.sqrt(xy[0] ** 2 + xy[1] ** 2)
    azimuth = np.arctan(np.abs(xy[0] / xy[1])) * 180 / np.pi
    if xy[0] > 0:
        if xy[1] > 0:
            azimuth = 180 - azimuth
    else:
        if xy[1] < 0:
            azimuth = 360 - azimuth
        else:
            azimuth = 180 + azimuth
    return speed, azimuth


if __name__ == '__main__':
    df = pd.read_csv(FNAME, delimiter=',')
    df_grouped = df.groupby(df['ObjectID']).mean().drop(
        labels='PointID', axis=1).to_numpy()
    for i, line in enumerate(df_grouped):
        xyspeed, std = calc_xyspeed(line)
        velocity, azimuth = calc_speed_azimuth(xyspeed)
        print(f'Velocity: {velocity * 3.6:.2f} km/h ({velocity * 3.6 / 1.852:.2f} knots) \
              ({(velocity * 3.6 / 1.852 - SPEED_GTS[i]) / SPEED_GTS[i] * 100: .2f}% \
                compared to the real speed {SPEED_GTS[i]} KNOTS or {SPEED_GTS[i] * 1.852: .2f} km/h), \
              Azimuth: {azimuth:.2f} degrees ({(azimuth - AZIMUTH_GTS[i]) / AZIMUTH_GTS[i] * 100: .2f}% \
                compared to the real speed {AZIMUTH_GTS[i]} DEG)')
