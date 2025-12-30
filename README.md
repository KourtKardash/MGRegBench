# MGRegBench

This repository provides the methods' implementation and dataset accompanying our paper: **MGRegBench: A Novel Benchmark Dataset with Anatomical Landmarks for Mammography Image Registration**.

Preprint available at: https://arxiv.org/abs/2512.17605

## Introduction

Robust and accurate registration of longitudinal mammograms is crucial for clinical tasks such as accurate tracking of lesions, monitoring tissue changes, and supporting radiologist decision-making. However, the field has long suffered from the lack of public, standardized benchmarks.  
To address this gap, we introduce **MGRegBench** — the first large-scale public dataset for 2D mammography registration that includes over 5,000 image pairs, with 100 containing manual anatomical landmarks and segmentation masks for rigorous evaluation. We also provide a comprehensive evaluation framework and benchmark results for a wide range of registration methods.

**Disclaimer.**

**This is a shortened version of the repository that retains the final structure and code style to provide a key overview of the work. To illustrate the data structure, we provide two data examples for each dataset in the train and evaluation splits. The methods are provided in their final code style, though we have omitted key parts containing our reimplementations and adaptations of other methods. The full version will be publicly available upon paper acceptance and can be provided upon request (skrasnova005@gmail.com).**

## Methods

We benchmark a set of diverse registration methods on MGRegBench, spanning classical optimization-based approaches, deep learning architectures, and implicit neural representations. Below is the list of evaluated methods along with links to their original publications.

- Affine <a href="#ref1">[1]</a>
- ANTs (SyN) <a href="#ref2">[2]</a>
- Curvilinear coordinates based method <a href="#ref3">[3]</a>
- IDIR (INR) <a href="#ref4">[4]</a>
- VoxelMorph <a href="#ref5">[5]</a>
- TransMorph <a href="#ref6">[6]</a>
- MammoRegNet <a href="#ref7">[7]</a>

## Dataset Structure 

The MGRegBench dataset is organized into _train_ and _evaluation_ directories for method development and performance assessment, respectively.

Data in both directories are grouped by source datasets (_INBreast_, _KAU-BCMD_) within patient-specific folders named with anonymized IDs. Each patient folder contains 2 (or more in some cases) images: PNG (and original DICOM) for INBreast and JPG for KAU-BCMD.

Due to licensing, the RSNA dataset is not redistributed. Instead, we provide a script that, when run on a local copy of the official RSNA dataset, extracts the relevant image pairs and integrates them into the MGRegBench structure in PNG format (preserving original DICOMs), ensuring reproducibility while complying with data agreements.

Ground-truth breast segmentation masks are in the _evaluation-masks_ directory, mirroring the image structure. Masks for all 3 source datasets are stored in PNG format.

Expert anatomical landmark annotations for the evaluation set are in 3 separate XML files:\
(1) landmarks by the first radiologist on the first image of each pair (file name is _moving_landmarks.xml_),\
(2) corresponding landmarks by the same radiologist on the second image (file name is _fixed_landmarks_1.xml_), and\
(3) independent annotations by a second radiologist on the second image for inter-observer validation (file name is _fixed_landmarks_2.xml_).

Annotation protocols for both phases, including an example of what constitutes corresponding landmark locations, are provided in the [Protocol.pdf](https://github.com/KourtKardash/MGRegBench/blob/main/Protocol.pdf) file.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KourtKardash/MGRegBench.git
   cd MGRegBench
   pip install -r requirements.txt
2. **Unpack the dataset**:
   ```bash
   unzip Dataset/MGRegBench.zip -d Dataset/MGRegBench
3. **RSNA dataset preparation**:
   
   - Download the RSNA dataset according to the instructions on the [link](https://kaggle.com/competitions/rsna-breast-cancer-detection)
   - Run the script:
      ```bash
      python Dataset/prepare_rsna_for_mgregbench.py --rsna-root /path/to/RSNA/dataset/folder
      ```

4. **Run registration methods**.
   Each method is implemented in its own subdirectory under Methods/. Use the corresponding scripts to run them:
   - Affine, classical affine registration (Ants). The Ants framework is known for its instability. To reproduce the results from the article, you need to run the method in 1 thread using `os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"`. To reproduce the results from the article from scratch, run the following script.
     ```bash
     python Methods/Affine/run_from_scratch.py
     ```
     However, the program will run much longer in one thread. Therefore, I have saved all the necessary transformations. In the evaluation part of the dataset, in each folder corresponding to each patient, in addition to mammograms, there is also an Affine transformation matrix (file name is _affine.mat_). You can run the following script, it will simply apply the existing transformations and reproduce the results from the article, which will work very quickly. Further DL methods in this benchmark use exactly these saved transformations.
     ```bash
     python Methods/Affine/run.py
     ```
   - SyN (Ants). Just like above, you can reproduce the results from the article from scratch using a script, but the execution will take longer, since the script must be run in a single thread for reproducibility of the results.
     ```bash
     python Methods/SyN/run_from_scratch.py
     ```
     Therefore, I also provide the deformation fields obtained using the SyN method for each pair of images from the evaluation set (file name is _warp.nii.gz_). The following script will apply the existing Affine and non-rigid transformations.
     ```
     python Methods/SyN/run.py
     ```
   - Curvilinear coordinates based method.
     ```bash
     python Methods/CurCoords/run.py
   - IDIR (INR).
     ```bash
     python Methods/IDIR/run.py

   - VoxelMorph.
     
      Training:
        ```bash
        python Methods/voxelmorph/scripts/torch/train.py
        ```
      Inference:
        ```bash
        python Methods/voxelmorph/inference.py
        ```
   - TransMorph.
     
      Training:
        ```bash
        python Methods/transmorph/train_TransMorph.py
        ```
      Inferece:
        ```bash
        python Methods/transmorph/inference.py
        ```
   - MammoRegNet.
     
      Training:
        ```bash
        python -m Methods.MammoRegNet.train.main_train_mammoregnet
        ```
      MammoRegNet inference with affine preprocessing using Ants:
        ```bash
        python -m Methods.MammoRegNet.evaluate.inference
        ```
      Original MammoRegNet without preprocessing:
        ```bash
        python -m Methods.MammoRegNet.evaluate.inference_original
        ```

## References

- <a id="ref2">[1]</a> Brian B Avants, Nick Tustison, Gang Song, et al. Advanced normalization tools (ants). Insight j, 2(365):1–35, 2009
- <a id="ref1">[2]</a> B.B. Avants, C.L. Epstein, M. Grossman, and J.C. Gee. Symmetric Diffeomorphic Image Registration with Cross-correlation: Evaluating Automated Labeling of Elderly and Neurodegenerative Brain. Medical Image Analysis, 12(1): 26–41, 2008
- <a id="ref3">[3]</a> Mohamed Abdel-Nasser, Antonio Moreno, and Domenec Puig. Temporal Mammogram Image Registration Using Optimized Curvilinear Coordinates. Computer Methods and Programs in Biomedicine, 127:1–14, 2016
- <a id="ref4">[4]</a> Jelmer M. Wolterink, Jesse C. Zwienenberg, and Christoph Brune. Implicit Neural Representations for Deformable Image Registration. In Proceedings of the 5th International Conference on Medical Imaging with Deep Learning (MIDL), pages 1349–1359. PMLR, 2022
- <a id="ref5">[5]</a> Guha Balakrishnan, Amy Zhao, Mert Sabuncu, John Guttag, and Adrian V. Dalca. VoxelMorph: A Learning Framework for Deformable Medical Image Registration. IEEE Transactions on Medical Imaging, 38:1788–1800, 2019
- <a id="ref6">[6]</a> Junyu Chen, Eric C. Frey, Yufan He, William P. Segars, Ye Li, and Yong Du. TransMorph: Transformer for Unsupervised Medical Image Registration. Medical Image Analysis, 78:102615, 2022
- <a id="ref7">[7]</a> Solveig Thrun, Stine Hansen, Zijun Sun, Nele Blum, Suaiba A Salahuddin, Kristoffer Wickstrøm, Elisabeth Wetzer, Robert Jenssen, Maik Stille, and Michael Kampffmeyer. Reconsidering Explicit Longitudinal Mammography Alignment for Enhanced Breast Cancer Risk Prediction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 495–505. Springer, 2025
