Dynamic Mixture of Multimodal Experts for Cancer diagnosis and prognosis
===========
Author list:
to complete [xxx](xxxx)

<img src="framework.png" width="1000px" align="center" />

**Abstract:** Multimodal fusion of pathological images and genomic data plays a pivotal role in cancer diagnosis and prognosis prediction. However, the simultaneous utilization of these two modalities presents two significant challenges: substantial heterogeneity between modalities and diverse interaction patterns across patient samples. Traditional fusion methods typically employ static, fixed frameworks that cannot dynamically adjust fusion strategies based on sample characteristics, thus limiting their ability to adapt to modality heterogeneity and individual variations. To address these challenges, we propose a Dynamic Mixture of Multimodal Experts Model (DynMoME) for cancer diagnosis and prognosis prediction. Our model implements multiple rounds of alternating encoding of pathological and genomic features through Dynamic Mixture of Multimodal Experts (DynMoME) layers. Each layer incorporates multiple fusion experts with relatively independent specializations, achieving multimodal fusion through the selection of optimal expert combinations. Notably, we introduce an innovative top-k_any gating mechanism that automatically determines the required number of experts for each sample and flexibly adjusts fusion strategies based on data-specific characteristics. This design not only captures rich cross-modal interactions through hierarchical fusion but also enables adaptive expert selection for optimal modality integration. Extensive experiments on five TCGA cancer benchmark datasets demonstrate that DynMoME achieves superior performance in both survival prediction and cancer grading tasks compared to state-of-the-art methods.

## Updates:
* 2025 Apr. 10th: Uploaded the codes and updated README.

## Pre-requisites:
* Linux (Tested on Ubuntu 20.04)
* NVIDIA GPU (Tested on A100)

## Dependencies:
```bash
torch
torchvision
scikit-survival
numpy
h5py
scipy
scikit-learning
pandas
nystrom_attention
admin_torch
```

## Preprocessing
Thanks to the great work of [CLAM](https://github.com/mahmoodlab/CLAM/tree/master). In this step, we used codes of [CLAM](https://github.com/mahmoodlab/CLAM/tree/master). Please refer to their original repository on how to process your WSIs into features.

The data used for training, validation and testing are expected to be organized as follows:
```bash
DATA_ROOT_DIR/
    ├──DATASET_1_DATA_DIR/
        └── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
    ├──DATASET_2_DATA_DIR/
        └── pt_files
                ├── slide_a.pt
                ├── slide_b.pt
                └── ...
    └──DATASET_3_DATA_DIR/
        └── pt_files
                ├── slide_i.pt
                ├── slide_ii.pt
                └── ...
    └── ...
```

### Genomic Data and task labels
The genomic data and survival labels are from [MCAT](https://github.com/mahmoodlab/MCAT/tree/master), with grade label organized from cBioPortal[https://www.cbioportal.org/]. you can directly use the preprocessed genomics and labels [here](./datasets_csv).

### GPU Training on TCGA-GBMLGG Dataset for survival prediction Using Our Default Settings
Run the following script:
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --model_type dynmome --apply_sig --results_dir ./path_to_result_dir/ --dataset gbmlgg --mome_gating_network CosMLP --data_root_dir /path_to_WSI_feature_root_dir/ --task_type survival
```
Simply replace the `survival` and `gbmlgg` with your desired task (like `grade`) and datasets to conduct experiments on different tasks and datasets.

## Issues
- Please report all issues on GitHub.

## Acknowledgement
This repository is built upon [CLAM](https://github.com/mahmoodlab/CLAM/tree/master), [MCAT](https://github.com/mahmoodlab/MCAT/tree/master), [MOTCat](https://github.com/Innse/MOTCat) and [MoME](https://github.com/BearCleverProud/MoME). Thanks again for their great works!

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](/to_complete/):

```
to complete
```
