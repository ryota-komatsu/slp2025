#!/bin/sh

dataset_root=${1:-data}

wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_audio_development.7z
wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_captions_development.csv
wget -P ${dataset_root} https://zenodo.org/records/4783391/files/clotho_metadata_development.csv