
## <img src="https://github.com/liuzy0708/liuzy0708.github.io/blob/main/assets/img/SL.png" width="5%" alt="" align=center /> Awesome_OL: A General Toolkit for Online Learning Approaches

Welcome to Awesome_OL, your comprehensive toolkit for online learning strategies and classifiers! This repository provides a collection of state-of-the-art strategies and classifiers for online active learning (OAL) and online semi-supervised learning (OSSL). Whether you're a researcher, practitioner, or enthusiast in machine learning, this toolkit offers valuable resources and implementations to enhance your projects.

### OAL Strategies:

Explore a variety of online active learning strategies located in the **OAL_strategies** folder:
#### Recent Progress:
| Strategy          | Description                                                                                                           | Reference            | Code Source    | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| CogDQS            | A dual-query strategy using Ebbinghaus’s law of human memory cognition, enabling experts to annotate the most representative samples. It employs a fixed uncertainty strategy for auxiliary judgment. | [Paper](https://ieeexplore.ieee.org/abstract/document/9492291)      |    NA   | 2023 | IEEE Transactions on Neural Networks and Learning Systems
| DSA-AI            |  A dynamic submodular-based learning strategy with activation interval for imbalanced drifting streams, which aims to address the challenges posed by concept drifts in nonstationary environments.| [Paper](https://ieeexplore.ieee.org/abstract/document/10195233/)     |   [Link](https://github.com/liuzy0708/DSLS-Demo)    | 2024 | IEEE Transactions on Neural Networks and Learning Systems
| MTSGQS            | A memory-triggered submodularity-guided query strategy that evaluates sample value through residual analysis and limited retraining, effectively addressing imbalanced data stream issues. | [Paper](https://ieeexplore.ieee.org/abstract/document/10105849)       |  NA     | 2023 | IEEE Transactions on Intelligent Transportation Systems

### OAL Classifiers:
| Classifier          | Description                                                                                                           | Reference            | Code Source    | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| ROALE-DI          |  A reinforcement online active learning ensemble for drifting imbalanced data streams, which combines uncertainty and imbalance strategies to evaluate sample value. | [Paper](https://ieeexplore.ieee.org/abstract/document/9204849)     |   [Link](https://github.com/saferhand/ROALE-DI)    |  2022 | IEEE Transactions on Knowledge and Data Engineering
| OALE         | An online active learning ensemble framework for drifting data streams based on a hybrid labeling strategy that includes an ensemble classifier and active learning strategies  | [Paper](https://ieeexplore.ieee.org/abstract/document/8401336)    |   NA   |  2019 | IEEE Transactions on Neural Networks and Learning Systems
#### Baseline Strategies:

| Strategy          | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| RS  | Random Sampling (RS) serves as a simple baseline for active learning, randomly selecting data samples for labeling without considering their informativeness. |  NA      |    NA    | NA | NA
| US_fix  | Uncertainty Sampling with Fixed Threshold (US_fix) selects samples with uncertainty scores exceeding a fixed threshold for labeling, effectively targeting uncertain regions of the data space. | [Paper](https://ieeexplore.ieee.org/abstract/document/6414645)      |   NA     | 2014 | IEEE Transactions on Neural Networks and Learning Systems
| US_var  | Uncertainty Sampling with Variable Threshold (US_var) dynamically adjusts the uncertainty threshold based on model confidence and dataset characteristics, offering improved sample selection flexibility and performance in dynamic environments. | [Paper](https://ieeexplore.ieee.org/abstract/document/6414645)      |  NA   | 2014 | IEEE Transactions on Neural Networks and Learning Systems

### OSSL Classifiers:

Discover online semi-supervised learning classifiers in the **OSSL_strategies** folder:
#### Recent Progress:
| Classifier        | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| OSSBLS            | An online semi-supervised BLS method with a loss function incorporating static anchor points. | [Paper](https://ieeexplore.ieee.org/abstract/document/9314231)       |   NA    | 2021 | IEEE Transactions on Industrial Informatics
| ISSBLS            | An online semi-supervised BLS method that ignores the relationship between historical data.  | [Paper](https://ieeexplore.ieee.org/abstract/document/9314231)       |  NA     | 2021 |IEEE Transactions on Industrial Informatics

#### Baseline Strategies:
| Classifier        | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| SOSELM            | A classic online semi-supervised learning method based on extreme learning machines. | [Paper](https://www.sciencedirect.com/science/article/pii/S0925231215011212)       |    NA  | 2016 |Neurocomputing

### Supervised Classifiers:

Find various online learning classifiers in the **classifer** folder:
#### Baseline Strategies:
| Classifier        | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| OLI2DS            | An online learning algorithm for imbalanced data streams that tackles dynamically evolving feature spaces and imbalances and empirical risk minimization using dynamic cost strategies. | [Paper](https://ieeexplore.ieee.org/abstract/document/10058539)     |   [Link](https://github.com/youdianlong/OLI2DS)    | 2023 | IEEE Transactions on Knowledge and Data Engineering
| DES               | An online ensemble learning method designed to adapt to data drift in streams with class imbalance, employing an improved Synthetic Minority Oversampling TEchnique (SMOTE) concept. | [Paper](https://ieeexplore.ieee.org/abstract/document/9802893)      |   [Link](https://github.com/Jesen-BT/DES-ICD)   | 2024 | IEEE Transactions on Neural Networks and Learning Systems
| BLS-W             | An online learning method based on the standard BLS architecture, utilizing the Sherman–Morrison Woodbury formula for incremental updates. | [Paper](https://ieeexplore.ieee.org/abstract/document/10375819)      |   [Link](https://github.com/liuzy0708/DMI-LS-Demo)   |   2023* | IEEE Transactions on Cybernetics
| IWDA             | A novel learner-agnostic algorithm for drift adaptation, which estimates the joint probability density of input and target for the incoming data. As soon as drift is detected, it retrains the learner using importance-weighted empirical risk minimization. | [Paper](https://ieeexplore.ieee.org/abstract/document/10105220)    |   [Link](https://github.com/SirPopiel/IWDA)   |   2023* | IEEE Transactions on Neural Networks and Learning Systems
| ACDWM             | An adaptive chunk-based incremental learning method is proposed for handling imbalanced streaming data with concept drift, utilizing statistical hypothesis tests to dynamically select chunk sizes for assessing classifier stability. | [Paper](https://ieeexplore.ieee.org/document/8924892)       |   [Link](https://github.com/jasonyanglu/ACDWM)    | 2020 | IEEE Transactions on Neural Networks and Learning Systems
| ARF               | An advanced online ensemble learning method that addresses changing data streams by integrating effective resampling methods and adaptive operators with ADWIN. | [Paper](https://link.springer.com/article/10.1007/s10994-017-5642-8)       |  [Link](https://github.com/scikit-multiflow/scikit-multiflow) | 2017 | Machine Learning
| SRP               | An ensemble method specially adapted to stream classification which combines random subspaces and online bagging. | [Paper](https://ieeexplore.ieee.org/document/8970784)       |   [Link](https://github.com/scikit-multiflow/scikit-multiflow)   | 2019 | ICDM

#### Baseline Strategies:
| Classifier        | Description                                                                                                           | Reference            | Code Source        | Year  | Journal/Conference|
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|--------------------|--------------------|
| OSELM               | An online sequential extreme learning machine model, which tries to iteratively update with the structure of extreme learning machines without the drift detection and adaption technique. | [Paper](https://ieeexplore.ieee.org/abstract/document/4012031)       |   [Link](https://github.com/leferrad/pyoselm)   | 2006 | IEEE Transactions on Neural Networks


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
| IWDA    |              |     ✅       |    ✅     |       ✅      |               ✅           |    ✅      |
| DES   |              |       ✅     |    ✅     |             |              ✅            |    ✅      |
| ACDWM   |              |     ✅       |    ✅     |             |              ✅            |    ✅      |
| SRP  |              |        ✅    |     ✅    |      ✅      |               ✅           |     ✅     |
| ARF  |              |       ✅     |     ✅    |      ✅      |              ✅            |     ✅     |

### Datasets:

The **datasets** folder contains .csv files structured with attributes, headers, and labels, catering to the needs of various strategies and classifiers.

### Visualization:

The **visualization** folder contains implementations for visualizing metrics such as accuracy (acc), macro F1 score, and other relevant performance measures.  

<img width="800" alt="stream" src="https://github.com/songqiaohu/pictureandgif/blob/main/Results_acc_LinearAbrupt-1.png?raw=true">

### Utility:

- **utils.py**: This component file serves as the interface between classifiers and strategies, facilitating seamless interaction within the toolkit.


### Implementation:

The specific implementations are encapsulated into a unified form. Further technical details and improvements can be explored within each strategy or classifier.

## Environment Setup:

Before using this library, please ensure that you have the following essential packages and their corresponding versions installed.

<center>

| Package            | Version       |
|--------------------|---------------|
| numpy              | 1.21.6        |
| matplotlib         | 3.2.2         |
| scikit-learn       | 0.22.1        |
| scikit-multiflow   | 0.5.3         |
| pandas             | 1.2.3         |
| scipy              | 1.7.3         |

</center>


Alternatively, for your convenience, you can set up the required environment by running the following command:

```
conda env create -f env.yml
```


## References:

Explore related resources and inspiration at:

- [GitHub - deep-active-learning](https://github.com/ej0cl6/deep-active-learning)
- [GitHub - scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow)

---

## Note

We hope this toolkit serves as a valuable asset in your online learning endeavors. Our team at the THUFDD Research Group, led by Prof. Xiao He and Prof. Donghua Zhou in the Department of Automation at Tsinghua University, is dedicated to fostering innovation and excellence in machine learning for industrial applications.

Your feedback, questions, and contributions are invaluable to us. Whether you have suggestions for improvements, encounter issues, or wish to collaborate on enhancements, we welcome your participation. Together, we can continue to refine and expand this toolkit to empower researchers, practitioners, and enthusiasts in the field.

Please feel free to reach out to us via email with [Zeyi Liu](mailto:liuzy21@mails.tsinghua.edu.cn) and [Songqiao Hu](mailto:hsq23@mails.tsinghua.edu.cn). Here's to a fruitful learning journey!

