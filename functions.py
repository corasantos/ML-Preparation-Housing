# My functions and classes

import os
import tarfile
from six.moves import urllib

def fetch_housing_data(housing_url=None, housing_path=None):

    if housing_path & housing_url:

        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)

        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

    else:
        print("Input Data URL source and target dir.")