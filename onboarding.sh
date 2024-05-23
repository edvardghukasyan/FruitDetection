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
            curl -o "fruits-360-original-size.zip" "https://storage.googleapis.com/kaggle-data-sets/5857/2609027/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240523%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240523T144154Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1b92039ef7a17349d999f58b5d26876018db7ea0f000c92daff8177fd6149eeffe178573ada0a90d6eadc0ccef1c7f936e8d939ac1908e20e63e84b6979e39296a61ca94bc2e82c2a6cd370679d78a3d1a0b453642b1f6ed57d1dc48eab10d8597b95da8e0798a946aa353e9f2f1311a9f262f448c807ae945546c8b99df3013a063d047940c8b4777697cd4f80160cd075cc22e9e7024d109526cc6dc6fcad6aadd9468217373c5f9975d1049ee97731e99a7a354b702ae6fe26cddcdd9d84b9de9706dfb79d2e20918aee7923465aa3778ae9983cdfc55894d393ac7ec95a40090e5dd84cd8f94ce94c2fac86c56b4237e7da5c6e951a368f18010adf949e8"
        fi
        yes | unzip fruits-360-original-size.zip && echo "Data successfully extracted into $data_dir"
    fi

    echo "Starting data merging..."
    cd preprocessing && python3 fruit_merger.py && cd .. && rm -rf $data_dir && rm -rf "$data_dir.zip"

fi

preprocessed_data_dir="fruits360_processed"

if [ -d "$preprocessed_data_dir" ]; then
    echo "Preprocessed data directory exists. Skipping data preprocessing."
else
    echo "Starting data preprocessing..."
    cd preprocessing && python3 fruit_processor.py && cd ..
fi
