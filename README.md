## Awesome_OL: Online Learning Toolkit

Welcome to Awesome_OL, your comprehensive toolkit for online learning strategies and classifiers! This repository provides a collection of state-of-the-art strategies and classifiers for online active learning (OAL) and online semi-supervised learning (OSSL). Whether you're a researcher, practitioner, or enthusiast in machine learning, this toolkit offers valuable resources and implementations to enhance your projects.

### OAL Strategies:

Explore a variety of online active learning strategies located in the **OAL_strategies** folder:

| Strategy          | Description                                                                                                           | Reference            | Code Source    | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| ROALE-DI          |  A reinforcement online active learning ensemble for drifting imbalanced data streams, which combines uncertainty and imbalance strategies to evaluate sample value. | [Paper]([1])     |       |  2023 | IEEE Transactions on Knowledge and Data Engineering
| CogDQS            | A dual-query strategy using Ebbinghaus’s law of human memory cognition, enabling experts to annotate the most representative samples. It employs a fixed uncertainty strategy for auxiliary judgment. | [Paper]([3])       |       | 2023 | IEEE Transactions on Neural Networks and Learning Systems
| DSA-AI            |  A dynamic submodular-based learning strategy with activation interval for imbalanced drifting streams, which aims to address the challenges posed by concept drifts in nonstationary environments.| [Paper]([2])       |   [Link](https://github.com/liuzy0708/DSLS-Demo)    | 2023 | IEEE Transactions on Neural Networks and Learning Systems
| MTSGQS            | A memory-triggered submodularity-guided query strategy that evaluates sample value through residual analysis and limited retraining, effectively addressing imbalanced data stream issues. | [Paper]([4])       |       | 2023 | IEEE Transactions on Intelligent Transportation Systems

### Baseline Strategies:

| Strategy          | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| Random Sampling (RS)  | Random Sampling (RS) serves as a simple baseline for active learning, randomly selecting data samples for labeling without considering their informativeness. |  NA      |        |
| Uncertainty Sampling with Fixed Threshold (US_fix)  | Uncertainty Sampling with Fixed Threshold (US_fix) selects samples with uncertainty scores exceeding a fixed threshold for labeling, effectively targeting uncertain regions of the data space. | [Paper]([5])       |        |
| Uncertainty Sampling with Variable Threshold (US_var)  | Uncertainty Sampling with Variable Threshold (US_var) dynamically adjusts the uncertainty threshold based on model confidence and dataset characteristics, offering improved sample selection flexibility and performance in dynamic environments. | [Paper]([5])       |     |

### OSSL Classifiers:

Discover online semi-supervised learning classifiers in the **OSSL_strategies** folder:

| Classifier        | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| OSSBLS            |  | [Paper]([11])       |      | 2021 | IEEE Transactions on Industrial Informatics
| ISSBLS            |   | [Paper]([11])       |      | 2021 |IEEE Transactions on Industrial Informatics
| SOSELM            |  | [Paper]([12])       |      | 2016 |Neurocomputing

### OAL Classifiers:

Explore various online active learning classifiers in the **OAL_classifiers** folder:

| Classifier        | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| OAL_Classifier1   |   |  | [Paper]([13])       |       | 2023 | Journal/Conference Name
| OAL_Classifier2   |   |  | [Paper]([14])       |       | 2023 | Journal/Conference Name
| OAL_Classifier3   |   |  | [Paper]([15])       |       | 2023 | Journal/Conference Name

### Supervised Classifiers:

Find various online learning classifiers in the **classifer** folder:

| Classifier        | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| OLI2DS            |  | [Paper]([6])       |       | 2023 | IEEE Transactions on Knowledge and Data Engineering
| DES               | | [Paper]([7])       |      | 2022 | IEEE Transactions on Neural Networks and Learning Systems
| BLS-W             | | [Paper]([2])       |      |   2022 | IEEE Transactions on Neural Networks and Learning Systems
| ACDWM             | | [Paper]([8])       |       | 2020 | IEEE Transactions on Neural Networks and Learning Systems
| SRP               |  | [Paper]([9])       |      | 2019 | ICDM
| ARF               |  | [Paper]([10])       |  | 2017 | Machien Learning

The following details are summarized for such implemented methods:

| Method  | OAL Strategy | Classifier | Binary Classification | Multi-class Classification | Concept Drift Adaptation | Ensemble |
|---------|:------------:|:----------:|:-------:|:-----------:|:------------------------:|:--------:|
| ROALE-DI |      ✅       |     ✅      |    ✅    |      ✅      |            ✅             |    ✅     |
| CogDQS  |      ✅       |            |    ✅     |      ✅      |              ✅            |          |
| DSA-AI  |      ✅       |            |     ✅    |      ✅      |             ✅             |          |
| MTSGQS  |      ✅       |            |    ✅     |      ✅      |            ✅              |          |
| RS      |      ✅       |            |     ✅    |      ✅      |                          |          |
| US-fix  |      ✅       |            |    ✅     |      ✅      |                          |          |
| US-var  |      ✅       |            |     ✅    |      ✅      |                          |          |
| OLI2DS  |              |     ✅       |    ✅     |             |               ✅           |          |
| DES   |              |       ✅     |    ✅     |             |              ✅            |    ✅      |
| ACDWM   |              |     ✅       |    ✅     |             |              ✅            |    ✅      |
| SRP  |              |        ✅    |     ✅    |      ✅      |               ✅           |     ✅     |
| ARF  |              |       ✅     |     ✅    |      ✅      |              ✅            |     ✅     |

### Datasets:

The **datasets** folder contains .csv files structured with attributes, headers, and labels, catering to the needs of various strategies and classifiers.

### Visualization:

The **visualization** folder contains implementations for visualizing metrics such as accuracy (acc), macro F1 score, and other relevant performance measures.

### Utility:

- **utils.py**: This component file serves as the interface between classifiers and strategies, facilitating seamless interaction within the toolkit.


### Implementation:

The specific implementations are encapsulated into a unified form. Further technical details and improvements can be explored within each strategy or classifier.

## Environment Setup:



To set up the required environment, run the following command:

```
conda env create -f env.yml
```

## Prerequisites

To use this library, make sure you have the following packages installed:

| Package                  | Version       | Package                  | Version       | Package                  | Version       |
|--------------------------|---------------|--------------------------|---------------|--------------------------|---------------|
| brotlipy                 | 0.7.0         | m2w64-gmp                | 6.1.0         | python                   | 3.7.1         |
| ca-certificates          | 2023.08.22    | m2w64-libwinpthread-git  | 5.0.0.4634.697f757 | python-dateutil    | 2.8.2         |
| certifi                  | 2022.12.7     | matplotlib-base          | 3.2.2         | python_abi               | 3.7           |
| cffi                     | 1.15.1        | mkl                      | 2020.4        | pytz                     | 2023.3.post1 |
| charset-normalizer       | 2.0.4         | msys2-conda-epoch        | 20160418      | qdldl-python             | 0.1.5         |
| cryptography             | 39.0.1        | numpy                    | 1.21.6        | requests                 | 2.28.1        |
| cvxpy                    | 1.2.1         | openssl                  | 1.1.1w       | scikit-learn             | 0.22.1        |
| cvxpy-base               | 1.2.1         | osqp                     | 0.6.2.post0  | scikit-multiflow         | 0.5.3         |
| cycler                   | 0.11.0        | pandas                   | 1.2.3         | scipy                    | 1.7.3         |
| ecos                     | 2.0.10        | pip                      | 22.3.1        | scs                      | 3.2.0         |
| flit-core                | 3.6.0         | prettytable              | 3.5.0         | setuptools               | 59.8.0        |
| freetype                 | 2.10.4        | pycparser                | 2.21          | six                      | 1.16.0        |
| idna                     | 3.4           | pyopenssl                | 23.0.0        | sortedcontainers         | 2.4.0         |
| importlib-metadata       | 4.11.3        | pyparsing                | 3.1.1         | tornado                  | 6.2           |
| intel-openmp             | 2023.2.0      | pysocks                  | 1.7.1         | typing_extensions        | 4.4.0         |
| joblib                   | 1.3.2         | python                   | 3.7.1         | urllib3                  | 1.26.14       |
| kiwisolver               | 1.4.4         | python-dateutil          | 2.8.2         | vc                       | 14.2          |
| libblas                  | 3.9.0         | python_abi               | 3.7           | vs2015_runtime           | 14.27.29016   |
| libcblas                 | 3.9.0         | pytz                     | 2023.3.post1 | wcwidth                  | 0.2.5         |
| liblapack                | 3.9.0         | qdldl-python             | 0.1.5         | wheel                    | 0.38.4        |
| libpng                   | 1.6.39        | requests                 | 2.28.1        | win_inet_pton            | 1.1.0         |

## References:

Explore related resources and inspiration at:

- [GitHub - deep-active-learning](https://github.com/ej0cl6/deep-active-learning)

The referenced papers are:

- [1] H. Zhang, W. Liu and Q. Liu, "Reinforcement Online Active Learning Ensemble for Drifting Imbalanced Data Streams," in <em>IEEE Transactions on Knowledge and Data Engineering</em>, vol. 34, no. 8, pp. 3971-3983, 1 Aug. 2022.
- [2] Z. Liu and X. He, "Dynamic Submodular-Based Learning Strategy in Imbalanced Drifting Streams for Real-Time Safety Assessment in Nonstationary Environments," in <em>IEEE Transactions on Neural Networks and Learning Systems</em>, 2023.
- [3] S. Liu et al., "Online Active Learning for Drifting Data Streams," in <em>IEEE Transactions on Neural Networks and Learning Systems</em>, vol. 34, no. 1, pp. 186-200, Jan. 2023.
- [4] Z. Liu and X. He, "Real-Time Safety Assessment for Dynamic Systems With Limited Memory and Annotations," in <em>IEEE Transactions on Intelligent Transportation Systems</em>, vol. 24, no. 9, pp. 10076-10086, Sept. 2023.
- [5] I. Žliobaitė, A. Bifet, B. Pfahringer and G. Holmes, "Active Learning With Drifting Streaming Data," in <em>IEEE Transactions on Neural Networks and Learning Systems</em>, vol. 25, no. 1, pp. 27-39, Jan. 2014.
- [6] D. You et al., "Online Learning From Incomplete and Imbalanced Data Streams," in <em>IEEE Transactions on Knowledge and Data Engineering</em>, vol. 35, no. 10, pp. 10650-10665, 1 Oct. 2023.
- [7] B. Jiao, Y. Guo, D. Gong and Q. Chen, "Dynamic Ensemble Selection for Imbalanced Data Streams With Concept Drift," in <em>IEEE Transactions on Neural Networks and Learning Systems</em>, 2022.
- [8] Y. Lu, Y. -M. Cheung and Y. Yan Tang, "Adaptive Chunk-Based Dynamic Weighted Majority for Imbalanced Data Streams With Concept Drift," in <em>IEEE Transactions on Neural Networks and Learning Systems</em>, vol. 31, no. 8, pp. 2764-2778, Aug. 2020.
- [9] H. M. Gomes, J. Read and A. Bifet, "Streaming Random Patches for Evolving Data Stream Classification," <em>2019 IEEE International Conference on Data Mining</em> (ICDM), Beijing, China, 2019, pp. 240-249.
- [10] H. M. Gomes, et al. "Adaptive random forests for evolving data stream classification." in <em>Machine Learning</em> vol. 106, pp. 1469-1495, 2017.
- [11] X. Pu and C. Li, "Online Semisupervised Broad Learning System for Industrial Fault Diagnosis," in <em>IEEE Transactions on Industrial Informatics</em>, vol. 17, no. 10, pp. 6644-6654, Oct. 2021.
- [12] X. Jia, et al. "A semi-supervised online sequential extreme learning machine method." in <em>Neurocomputing</em> vol. 174, pp. 168-178, 2016.

---

## Note

We hope this toolkit serves as a valuable asset in your online learning endeavors. Our team at the THUFDD Research Group, led by Prof. Xiao He and Prof. Donghua Zhou in the Department of Automation at Tsinghua University, is dedicated to fostering innovation and excellence in machine learning for industrial applications.

Your feedback, questions, and contributions are invaluable to us. Whether you have suggestions for improvements, encounter issues, or wish to collaborate on enhancements, we welcome your participation. Together, we can continue to refine and expand this toolkit to empower researchers, practitioners, and enthusiasts in the field.

Please feel free to reach out to us via email at [mailto:liuzy21@mails.tsinghua.edu.cn]. Here's to a fruitful learning journey!

