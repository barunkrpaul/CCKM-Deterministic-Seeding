#!/bin/bash
# Fetch all 29 downloadable benchmark datasets (Iris/Digits/Wine/Breast-Cancer come from scikit-learn)
mkdir -p data && cd data
B="https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets"
CORE="tetra chainlink flame pathbased aggregation R15 D31"
EXT="s-set1 s-set2 s-set3 s-set4 birch-rg1 birch-rg2 jain spiral compound atom hepta lsun target twodiamonds wingnut engytime"
for f in $CORE $EXT; do curl -sL "$B/artificial/$f.arff" -o "$f.arff"; done
curl -sL "$B/real-world/ecoli.arff" -o ecoli.arff
curl -sL "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv" -o seeds.csv
curl -sL "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz" -o mnist-images.gz
curl -sL "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz" -o mnist-labels.gz
