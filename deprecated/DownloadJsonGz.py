'''
This file is for downloading JSON format ADS-B data from `adsbexchange`.
Deprecated Now.
'''

import gzip
import random
import urllib.request
import os
import time
from concurrent.futures import ProcessPoolExecutor


def download_task(url, cur_dir, fname, opener):
    urllib.request.install_opener(opener)
    print(f"Downloading {fname}")
    ffname = os.path.join(cur_dir, fname)
    (dfname, header) = urllib.request.urlretrieve(url + fname, ffname)
    try:
        with gzip.GzipFile(fileobj=open(ffname, 'rb'), mode='rb') as g:
            with open(ffname.replace(".gz", ""), "wb") as f:
                f.write(g.read())
    except Exception as e:
        print(ffname, e)
    time.sleep(random.random() / 10)


def download_all(url: str, cur_dir: str):
    fnames = set(os.listdir(cur_dir))

    headers = {'User-Agent',
               'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}
    opener = urllib.request.build_opener()
    opener.addheaders = [headers]
    urllib.request.install_opener(opener)
    task_list = [
        (url, cur_dir, f'{hh:02d}{mm:02d}{ss:02d}Z.json.gz', opener)
        for hh in range(24) for mm in range(60) for ss in range(0, 60, 5)
        if f'{hh:02d}{mm:02d}{ss:02d}Z.json.gz' not in fnames and
           f'{hh:02d}{mm:02d}{ss:02d}Z.json' not in fnames
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_pool:
        for tl in task_list:
            process_pool.submit(download_task, *tl)


if __name__ == "__main__":
    URL = r'https://samples.adsbexchange.com/readsb-hist/2023/04/01/'
    DIR = r'C:\\Users\\Iridescent\\Documents\\adsb_json\\20230401'
    download_all(URL, DIR)
    for fname in os.listdir(DIR):
        if ".gz" in fname:
            os.remove(os.path.join(DIR, fname))
