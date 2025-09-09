#!/bin/sh
cd /home/jguerrero/Desarrollo/rocf-mci-detection/training
python3 train_siamese_model_with_rocf_dataset.py --fold_index 0
python3 train_siamese_model_with_rocf_dataset.py --fold_index 1
python3 train_siamese_model_with_rocf_dataset.py --fold_index 2
python3 train_siamese_model_with_rocf_dataset.py --fold_index 3
python3 train_siamese_model_with_rocf_dataset.py --fold_index 4
python3 train_siamese_model_with_rocf_dataset.py --fold_index 5
python3 train_siamese_model_with_rocf_dataset.py --fold_index 6
python3 train_siamese_model_with_rocf_dataset.py --fold_index 7
python3 train_siamese_model_with_rocf_dataset.py --fold_index 8
python3 train_siamese_model_with_rocf_dataset.py --fold_index 9
python3 train_siamese_model_with_rocf_dataset.py --fold_index 10
python3 train_siamese_model_with_rocf_dataset.py --fold_index 11
python3 train_siamese_model_with_rocf_dataset.py --fold_index 12
python3 train_siamese_model_with_rocf_dataset.py --fold_index 13