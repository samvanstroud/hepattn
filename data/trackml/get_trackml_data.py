import tarfile
import zipfile
from pathlib import Path

import kaggle
import wget

if __name__ == "__main__":
    # download the detectors.zip file from the trackML competition
    kaggle.api.competition_download_file("trackml-particle-identification", "detectors.zip", path="data/trackml/")
    with zipfile.ZipFile("data/trackml/detectors.zip", mode="r") as zip_ref:
        zip_ref.extractall("data/trackml/")
    # get small sample of training data from codalab trackML competition
    raw_dir = Path("data/trackml/raw")
    raw_dir.mkdir(exist_ok=True, parents=True)
    url = "https://cernbox.cern.ch/remote.php/dav/public-files/Nf0zSvfJcshdhIP/training_sample01.tar"
    fname_tar = raw_dir / "training_sample01.tar"
    wget.download(url, out=str(fname_tar))
    with tarfile.open(fname_tar, "r") as tar_ref:
        tar_ref.extractall(raw_dir, filter="tar")
