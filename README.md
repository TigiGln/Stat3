![Pyton 3.9.5](https://img.shields.io/badge/python-3.9.5-blue.svg)

# M2 DLAD Stat3 Machine learning project

## Analyze
* This data collection corresponds to a portion of the PANCAN RNA-Seq(HiSeq) data
* This is a gene expression extraction of patients with different tumor types:
- BRCA
- PRAD
- LUAD
- KIRC
- COAD
* We count 800 samples (rows)
* 20,531 genetic expressions (columns)
* Each gene expression is measured by the Illumina HiSeq sequencing platform

* We made a model with a deep logistic regression with simulations of the cases where the model is perfectible

* Then a short analysis with a neural network to compare the optimal model

## Dependencies

### Miniconda
```{}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source .bashrc
conda create -n stat3
conda activate stat3
```

### Installation packages for analysis
```{}
conda install -c conda-forge pandas
conda install -c conda-forge scikit-learn
conda install -c conda-forge keras tensorflow
```

### data
* Please download the data to be analyzed at the following address:
https://archive.ics.uci.edu/ml/machine-learning-databases/00401/
* Please click on:
TCGA-PANCAN-HiSeq-801x20531.tar.gz
* Unzip data
```{}
tar czvf TCGA-PANCAN-HiSeq-801x20531.tar.gz
```
* clone the depot
```{}
git clone https://github.com/TigiGln/Stat3.git
```
* go to the clone directory
```{}
mkdir data
```
* copy the two unzipped csv files to this data directory


## Launch projet
```{}
python3 main.py
```


![Screenshot](outputs/Figure_1.png)
