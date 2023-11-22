## Reducing Histopathology Slide Magnification Improves the Accuracy and Speed of Ovarian Cancer Subtyping
<img src="CISTIB logo.png" align="right" width="240"/>

*Brief Summary* 

<img src="ABMILarchitecture-min.png" align="centre" width="900"/>

This repo was created as part of a submission to [ISBI 2024](https://biomedicalimaging.org/2024/).

## Hyperparameters
Final Hyperparamters Determined by Hyperparameter Tuning: 
| Magnif. | Learning Rate | Weight Decay | First Moment Decay | Second Moment Decay | Stability Parameter | Attention Layer Size | Dropout | Max Patches |
| :-------: | :-------------: | :------------: |:------------------:|:-------------------: | :-------------------: | :--------------------: | :-------: | :-----------: |
| 40x     | 1e-3          | 1e-4         |        0.95        | 0.99                | 1e-10               | 512                  | 0.6     | 50000       |
| 20x     | 5e-4          | 1e-4         |        0.99        | 0.99                | 1e-8                | 256                  | 0.7     | 10000       |
| 10x     | 5e-4          | 1e-4         |        0.8         | 0.99                | 1e-4                | 256                  | 0.6     | 1000        |
| 5x      | 5e-4          | 1e-6         |        0.95        | 0.999               | 1e-14               | 128                  | 0.6     | 400         |
| 2.5x    | 1e-3          | 1e-5         |        0.9         | 0.9999              | 1e-4                | 256                  | 0.7     | 40          |
| 1.25x   | 5e-4          | 1e-5         |        0.9         | 0.999               | 1e-14               | 256                  |    0.5  | 7           |

&nbsp;

Hyperparameters were tuned in 13 stages in which 1-3 individual hyperparameters were altered and the rest were frozen. All specific configurations can be accessed in the folder tuning_configs. The maximum number of epochs was set to 30 for the first 3 stages, 100 for stages 4-9, 150 for stages 10-13:
- Stage 1: Learning Rate, Max Patches
- Stage 2: First Moment Decay, Second Moment Decay
- Stage 3: Weight Decay, Dropout
- Stage 4: Learning Rate, Dropout, Max Patches
- Stage 5: Attention Layer Size
- Stage 6: Learning Rate, Dropout, Max Patches
- Stage 7: Stability Parameter
- Stage 8: Learning Rate
- Stage 9: Dropout
- Stage 10: Learning Rate, Dropout, Max Patches
- Stage 11: Weight Decay
- Stage 12: Max Patches
- Stage 13: First Moment Decay, Second Moment Decay


## Code Examples
The following code includes examples from every stage of pre-processing, hyperparameter tuning, and model validation.  

<details>
<summary>
Tissue region extraction
</summary>
We segmented tissue using saturation thresholding and extracted non-overlapping tissue regions which corresponded to 256x256 pixel patches at 40x (e.g. 512x512 for 20x, 1024x1024 for 10x). At this stage all images are still at 40x magnification, and only the patch size is changing:
  
``` shell
## 40x 256x256 patches for use in 40x experiments
python create_patches_fp.py --source "/mnt/data/Katie_WSI/edrive" --save_dir "/mnt/results/patches/ovarian_leeds_mag40x_patch256_DGX_fp" --patch_size 256 --step_size 256 --seg --patch --stitch 	
## 40x 8192x8192 patches for use in 1.25x experiments
python create_patches_fp.py --source "/mnt/data/Katie_WSI/edrive" --save_dir "/mnt/results/patches/ovarian_leeds_mag40x_patch8192_DGX_fp" --patch_size 8192 --step_size 8192 --seg --patch --stitch 	
``` 
</details>

<details>
<summary>
Feature extraction
</summary>
We then downsampled the patches to the experimental magnification and extracted [1,1024] features from each 256x256 patch using a ResNet50 encoder pretrained on ImageNet:
  
``` shell
## 40x (no downsampling needed)
python extract_features_fp.py --hardware DGX --model_type 'resnet50' --data_h5_dir "/mnt/results/patches/ovarian_leeds_mag40x_patch256_DGX_fp" --data_slide_dir "/mnt/data/Katie_WSI/edrive" --csv_path "dataset_csv/set_edrivepatches_ESGO_train_staging.csv" --feat_dir "/mnt/results/features/ovarian_leeds_resnet50_40x_features_DGX" --batch_size 32 --slide_ext .svs 
## 1.25x (32x downsampling)
python extract_features_fp.py --hardware DGX --custom_downsample 32 --model_type 'resnet50' --data_h5_dir "/mnt/results/patches/ovarian_leeds_mag40x_patch8192_DGX_fp" --data_slide_dir "/mnt/data/Katie_WSI/edrive" --csv_path "dataset_csv/set_edrivepatches_ESGO_train_staging.csv" --feat_dir "/mnt/results/features/ovarian_leeds_resnet50_1point25x_features_DGX" --batch_size 32 --slide_ext .svs 
```
</details>

<details>
<summary>
Hyperparameter tuning
</summary>

Tuning tuning tuning
``` shell
## 40x tuning first stage first fold
python main.py --tuning --hardware DGX --tuning_output_file /mnt/results/tuning_results/staging_only_resnet50_40x_firsttuning_bce_fold0.csv --min_epochs 0 --max_epochs 30 --early_stopping --num_tuning_experiments 1 --split_dir "esgo_staging_5fold_100" --k 1 --results_dir /mnt/results --exp_code staging_only_resnet50_40x_firsttuning_30epochs_bce_fold0 --subtyping --weighted_sample --bag_loss balanced_ce --no_inst_cluster --task ovarian_5class  --model_type clam_sb --subtyping --csv_path 'dataset_csv/ESGO_train_all.csv' --data_root_dir "/mnt/results/features" --features_folder "ovarian_leeds_resnet50_40x_features_DGX" --tuning_config_file tuning_configs/esgo_staging_resnet50_40x_config1.txt
## 1.25x tuning first stage first fold
python main.py --tuning --hardware DGX --tuning_output_file /mnt/results/tuning_results/staging_only_resnet50_1point25x_firsttuning_bce_fold0.csv --min_epochs 0 --max_epochs 30 --early_stopping --num_tuning_experiments 1 --split_dir "esgo_staging_5fold_100" --k 1 --results_dir /mnt/results --exp_code staging_only_resnet50_1point25x_firsttuning_30epochs_bce_fold0 --subtyping --weighted_sample --bag_loss balanced_ce --no_inst_cluster --task ovarian_5class  --model_type clam_sb --subtyping --csv_path 'dataset_csv/ESGO_train_all.csv' --data_root_dir "/mnt/results/features" --features_folder "ovarian_leeds_resnet50_1point25x_features_DGX" --tuning_config_file tuning_configs/esgo_staging_resnet50_1point25x_config1.txt
```

After running all folds for a given magnification and tuning stage, the results were summarised into a csv for analysis:

``` shell
## 1.25x
python combine_results.py --file_base_name "/mnt/results/tuning_results/staging_only_resnet50_1point25x_firsttuning_bce"
```

</details>



<details>
<summary>
Model training
</summary>
The best model from the 5-fold cross-validation experiment (as judged by averaged validation set cross-entropy loss across three repeats and five folds) was trained:
  
``` shell
python main.py --hardware DGX --max_patches_per_slide 15 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.0 --lr 0.0005 --reg 0.0001 --model_size hipt_smaller --split_dir "treatment_5fold_100" --k 5 --results_dir /mnt/results --exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
```
</details>

<details>
<summary>
Model evaluation
</summary>
The model was evaluated on the test sets of the five-fold cross validation with 100,000 iterations of bootstrapping:
  
``` shell
python eval.py --drop_out 0.0 --model_size hipt_smaller --models_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_s1 --save_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_bootstrapping --task treatment --model_type clam_sb --results_dir /mnt/results --data_root_dir "/mnt/data" --k 5 --features_folder "treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --csv_path 'dataset_csv/set_treatment.csv' 
python bootstrapping.py --num_classes 2 --model_names  treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_bootstrapping --bootstraps 100000 --run_repeats 1 --folds 5
```

The cross-validation results for this optimal HIPT-ABMIL model were as follows:

``` shell
 Confusion Matrix:
 [[ 76  49]
 [ 29 128]]

 average ce loss:  0.4858174402095372 (not bootstrapped)
 AUC mean:  [0.8206680412411297]  AUC std:  [0.02530094639907452]
 F1 mean:  [0.7659177381223935]  F1 std:  [0.02579712919409385]
 accuracy mean:  [0.7234604255319149]  accuracy std:  [0.02667653193254119]
 balanced accuracy mean:  [0.7117468943178861]  balanced accuracy std:  [0.026864606981070703]
```
</details>

<details>
<summary>
Challenge test set
</summary>

First, the test set images were pre-processed into pyramid svs files through the same approach as used for the training set images (though these originated as .bmp files rather than .svs files), for example:

``` shell
vips tiffsave "I:\treatment_data\2023MICCAI_testing_set\0.BMP" "I:\treatment_data\testpyramid_jpeg90compress\0.svs" --compression jpeg --Q 90 --tile --pyramid
```

Patches were selected (one per slide due to the size of these images) and features extracted:
``` shell
python create_patches_fp.py --source "../mount_i/treatment_data/testpyramid_jpeg90compress" --save_dir "../mount_outputs/extracted_mag20x_patch4096_fp_testset_updated_Q90" --patch_size 4096 --step_size 4096 --seg --patch --stitch --pad_slide --sthresh 15 --mthresh 5 --use_otsu --closing 200 --atfilter 8
python extract_features_fp.py --use_transforms 'HIPT' --model_type 'HIPT_4K' --data_h5_dir "../mount_outputs/extracted_mag20x_patch4096_fp_testset_updated_Q90" --data_slide_dir "../mount_i/treatment_data/testpyramid_jpeg90compress" --csv_path "dataset_csv/set_treatment_test.csv" --feat_dir "../mount_outputs/features/treatment_hipt4096_features_normalised_test_updated_Q90patches" --batch_size 1 --slide_ext .svs
```

The hyperparameters of the best-performing model on internal data was applied to create an ensemble of four models:
``` shell
python main.py --hardware DGX --max_patches_per_slide 15 --data_slide_dir "../mount_i/treatment_data/pyramid_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.0 --lr 0.0005 --reg 0.0001 --model_size hipt_smaller --split_dir "treatment_submission_folds" --k 4 --results_dir results --exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_4fold_7525test --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment_plus_test.csv' --data_root_dir "../mount_outputs/features/" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
```

Finally, predictions were made on the TMA challenge test set, with the median of these predictions submitted for the challenge:
``` shell
python eval.py --drop_out 0.0 --model_size hipt_smaller --models_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_4fold_7525test_s1 --save_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_4fold_7525test_Q90patchestest_bootstrapping --task treatment --model_type clam_sb --results_dir results --data_root_dir "../mount_outputs/features/" --k 4 --features_folder "treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --csv_path 'dataset_csv/set_treatment_plus_test.csv'
```
</details>



## Reference
This code is an extension of our [previous repository](https://github.com/scjjb/DRAS-MIL), which itself was forked from the [CLAM repository](https://github.com/mahmoodlab/CLAM) with corresponding [paper](https://www.nature.com/articles/s41551-020-00682-w). This repository and the original CLAM repository are both available for non-commercial academic purposes under the GPLv3 License.
