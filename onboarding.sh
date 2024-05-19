#!/bin/bash

source /opt/anaconda3/etc/profile.d/conda.sh
conda env create
conda activate fruit_detection

merged_data_dir="fruits360_merged"
if [ -d "$merged_data_dir" ]; then
    echo "Merged data directory exists. Skipping data merge."
else
    data_dir="fruits-360-original-size"

    if [ -d "$data_dir" ]; then
        echo "Data directory already exists. Skipping download."
    else
        echo "Data directory does not exist."
        if [ -e "$data_dir.zip" ]; then
            echo "Zip file does exist."
        else
            echo "Zip file does not exist. Downloading zipped data..."
            curl -o "fruits-360-original-size.zip" "https://storage.googleapis.com/kaggle-data-sets/5857/2609027/compressed/fruits-360-original-size.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240519%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240519T103228Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=039e643a2dd07341f3fe2d2a7ea2cbf39feb6e79d4db9923ee8e291dc5b1feca1e67c729abbc16bb85101b4acba54bae44c0b57662f41a7c61fdd5ece1bc119594f455b80e5aa4bafc3f87ab5925235e0c47e8773d4870ce490a368918f63619c8adb397ab010c24ec4f1cdfe660d0adacae0e1b42a4d1656d1e28805e6b8c7e1cebe6881bf15a1844ab0c8322ca9ec5693a76fbaada50c265d34266060624e5f515f5d122013ec1e7e5a47b6122ebcad2b541fbcdbe1e8794cff80c892e8f335b57b66d00210cbb42eded1c1a394a529806b48cb3e0584e4a1cb21267f84098ac8dd835bb003a567b5258b2a7a45aa2575313da70ff0937c048e6e79cf7e8c2"
        fi
        yes | unzip fruits-360-original-size.zip && echo "Data successfully extracted into $data_dir"
    fi

    cd fruit_preprocessing && python3 fruit_merger.py && cd .. && rm -rf $data_dir && rm -rf "$data_dir.zip"

fi
