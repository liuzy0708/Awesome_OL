## Awesome_OL: Online Learning Toolkit

Welcome to Awesome_OL, your comprehensive toolkit for online learning strategies and classifiers! This repository provides a collection of state-of-the-art strategies and classifiers for online active learning (OAL) and online semi-supervised learning (OSSL). Whether you're a researcher, practitioner, or enthusiast in machine learning, this toolkit offers valuable resources and implementations to enhance your projects.

### Contents:

- **utils.py**: This component file serves as the interface between classifiers and strategies, facilitating seamless interaction within the toolkit.

### OAL Strategies:

Explore a variety of online active learning strategies located in the **OAL_strategies** folder:

| Strategy          | Description                                                                                                           | Reference            | Source Code        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| ROALE-DI          | Recent Object-Adaptive Learning with Dynamic Importance (ROALE-DI) dynamically adjusts the importance of different data samples based on their relevance to the evolving model, effectively maximizing model performance with limited labeled data. Introduced in the 2023 IEEE-TKDE.  | [Paper](link)       | [Link](link)       |
| CogDQS            | Cognitive Drift-Quelling Strategy (CogDQS) addresses the challenge of concept drift in online learning scenarios by detecting and mitigating cognitive drift, ensuring model stability and adaptability over time. Proposed in the 2023 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| DSA-AI            | Dynamic Sample Acquisition with Artificial Intelligence (DSA-AI) leverages advanced AI techniques to intelligently select data samples for labeling, dynamically adjusting the sample acquisition process based on model performance and data distribution. Discussed in the 2023 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| MTSGQS            | Multi-Task Sequential Gaussian Quadratic Sampling (MTSGQS) incorporates multi-task learning principles to enhance sample selection efficiency and model generalization by exploiting correlations among related tasks. Presented in the 2023 IEEE-TITS. | [Paper](link)       | [Link](link)       |

### Baseline Strategies:

| Strategy          | Description                                                                                                           | Reference            | Source Code        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| Random Sampling (RS)  | Random Sampling (RS) serves as a simple baseline for active learning, randomly selecting data samples for labeling without considering their informativeness. | [Paper](link)       | [Link](link)       |
| Uncertainty Sampling with Fixed Threshold (US_fix)  | Uncertainty Sampling with Fixed Threshold (US_fix) selects samples with uncertainty scores exceeding a fixed threshold for labeling, effectively targeting uncertain regions of the data space. | [Paper](link)       | [Link](link)       |
| Uncertainty Sampling with Variable Threshold (US_var)  | Uncertainty Sampling with Variable Threshold (US_var) dynamically adjusts the uncertainty threshold based on model confidence and dataset characteristics, offering improved sample selection flexibility and performance in dynamic environments. | [Paper](link)       | [Link](link)       |

### OSSL Classifiers:

Discover online semi-supervised learning classifiers in the **OSSL_strategies** folder:

| Classifier        | Description                                                                                                           | Reference            | Source Code        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| OSSBLS            | Online Semi-Supervised Binary Learning System (OSSBLS) is a powerful framework for semi-supervised learning in online settings, enhancing model robustness and scalability by leveraging both labeled and unlabeled data. Introduced in the 2021 IEEE-TII. | [Paper](link)       | [Link](link)       |
| ISSBLS            | Incremental Semi-Supervised Binary Learning System (ISSBLS) extends OSSBLS by incorporating incremental learning principles, achieving continuous improvement and adaptation to evolving data distributions. Presented in the 2021 IEEE-TII. | [Paper](link)       | [Link](link)       |
| SOSELM            | Self-Organizing Semi-Supervised Extreme Learning Machine (SOSELM) combines the advantages of self-organizing maps (SOMs) and extreme learning machines (ELMs) for semi-supervised learning, enhancing model generalization and scalability by exploiting the structure of unlabeled data. Described in the 2016 Neurocomputing. | [Paper](link)       | [Link](link)       |

### Supervised Classifiers:

Find various online learning classifiers in the **classifer** folder:

| Classifier        | Description                                                                                                           | Reference            | Source Code        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| OLI2DS            | Online Incremental Two-Dimensional Space Classifier (OLI2DS) is a novel approach for online learning in two-dimensional data spaces, achieving high accuracy and low computational complexity by efficiently updating model parameters and decision boundaries. Presented in the 2023 IEEE-TKDE. | [Paper](link)       | [Link](link)       |
| DES               | Dynamic Ensemble Selection (DES) is an ensemble learning technique designed for dynamic environments, adapting to changing data distributions and concept drift by dynamically selecting and combining base classifiers. Described in the 2022 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| BLS-W             | Balanced Label Space for Online Learning (BLS-W) addresses the challenge of imbalanced label distributions in online learning tasks by incorporating label balancing techniques, improving model fairness and generalization. Introduced in the 2021 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| ACDWM             | Adaptive Contextual Decision Weighting Model (ACDWM) is a flexible framework for contextual online learning, achieving adaptive model updating and robust performance in dynamic environments by dynamically adjusting decision weights based on contextual information. Discussed in the 2020 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| SRP               | Stream-based Random Projection (SRP) is a dimensionality reduction technique tailored for streaming data, reducing computational complexity and memory requirements by projecting high-dimensional data onto a lower-dimensional space. Presented in the 2019 ICDM. | [Paper](link)       | [Link](link)       |
| ARF               | Adaptive Random Forest (ARF) is a variant of the random forest algorithm designed for online learning tasks, achieving high accuracy and efficiency by adaptively updating decision trees and pruning outdated branches. Introduced in the 2017 Machine Learning. | [Paper](link)       | [Link](link)       |


### Datasets:

The **datasets** folder contains .csv files structured with attributes, headers, and labels, catering to the needs of various strategies and classifiers.

### Implementation:

The specific implementations are encapsulated into a unified form. Further technical details and improvements can be explored within each strategy or classifier.

## Environment Setup:

To set up the required environment, run the following command:

```
conda env create -f env.yml
```

## References:

Explore related resources and inspiration at:

- [GitHub - deep-active-learning](https://github.com/ej0cl6/deep-active-learning)
- 
Thank you for selecting Awesome_OL! We trust that this toolkit will prove to be a valuable resource in your online learning pursuits.

We are from the THUFDD Research Group, under the guidance of Prof. Xiao He and Prof. Donghua Zhou, situated within the Department of Automation at Tsinghua University.

Should you have any inquiries, feedback, or wish to contribute, we encourage you to engage with our community via [mailto:liuzy21@mails.tsinghua.edu.cn]. Here's to a fruitful learning journey!
