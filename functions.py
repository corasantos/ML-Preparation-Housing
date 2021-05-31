# My functions and classes

import os
import tarfile
from six.moves import urllib

def fetch_housing_data(housing_url='', housing_path=''):

    if len(housing_path) != 0 | len(housing_url) != 0:

        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)

        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

    else:
        print("Input Data URL source and/or target dir.")

