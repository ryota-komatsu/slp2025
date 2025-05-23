#!/bin/sh
# sudo apt install -y p7zip-full

dataset_root=${1:-data/clotho}

wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_audio_development.7z
wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_captions_development.csv
wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_metadata_development.csv

wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_audio_validation.7z
wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_captions_validation.csv
wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_metadata_validation.csv

cd ${dataset_root}
7z x clotho_audio_development.7z
7z x clotho_audio_validation.7z