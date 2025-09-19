# Mild Cognitive Impairment Detection from Rey-Osterrieth Complex Figure Copy Drawings using a Contrastive Loss Siamese Neural Network

![](https://img.shields.io/badge/language-Python-{green}.svg)
![](https://img.shields.io/badge/license-GNU-{yellowgreen}.svg)

This code repository is the official source code of the paper ["Mild Cognitive Impairment Detection from Rey-Osterrieth Complex Figure Copy Drawings using a Contrastive Loss Siamese Neural Network"](https://edatos.consorciomadrono.es/dataverse/rey) by [Juan Guerrero Martín et al.](http://www.simda.uned.es/)

## Requirements

Operating system: GNU/Linux Debian 13 (trixie=stable 2025-08-09)

Hardware environment: Intel(R) Core(TM) i7-7820X CPU @ 3.60Ghz, 32 GB RAM, NVIDIA GeForce GTX 1080 GPU.

Programming language: Python 3.8.12

Programming libraries: TensorFlow + Keras 2.4.1

Download the [augmented ROCFD528 (binary images)](https://doi.org/10.21950/MFHOO4) dataset. 

The default directory of the dataset is:
/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528_augmented/

Please note that the files data/dataset_information/rocf_reference.png (the ROCF model), data/dataset_information/rocfd528_info.csv (ROCFD528 labels), and data/dataset_information/rocfd528_augmented_info.csv (augmented ROCFD528 labels) are only available upon formal request.

## Usage

```
# 1. Choose your workspace and download our repository.
cd ${CUSTOMIZED_WORKSPACE}
git clone https://github.com/SIMDA-UNED/rocf-mci-detection.git

# 2. Enter the directory.
cd rocf-mci-detection

# 3. Convert our dataset into a pickle.
python utils/dataset_to_pickle.py

# 4. Execute any of our scripts.

Example:

cd training
python train_siamese_model_with_rocf_dataset.py
```

## Citations

If you find this code useful to your research, please cite our paper as the following bibtex:

```
TBD
```

## License

This project is licensed under the GNU General Public License v3.0.

## Funding

This research has been supported by an FPI-UNED-2021 scholarship.

## Contact

If you would have any discussion on this code repository, please feel free to send an email to Juan Guerrero Martín.  
Email: **jguerrero@dia.uned.es**