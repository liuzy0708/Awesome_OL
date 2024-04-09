## Awesome_OL: Online Learning Toolkit

Welcome to Awesome_OL, your comprehensive toolkit for online learning strategies and classifiers! This repository provides a collection of state-of-the-art strategies and classifiers for online active learning (OAL) and online semi-supervised learning (OSSL). Whether you're a researcher, practitioner, or enthusiast in machine learning, this toolkit offers valuable resources and implementations to enhance your projects.

### Contents:

- **utils.py**: This component file serves as the interface between classifiers and strategies, facilitating seamless interaction within the toolkit.

### OAL Strategies:

Explore a variety of online active learning strategies located in the **OAL_strategies** folder:

| Strategy          | Description                                                                                                           | Reference            | Code Source        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| ROALE-DI          | Proposed in the 2023 IEEE-TKDE.  | [Paper](link)       | [Link](link)       |
| CogDQS            | Proposed in the 2023 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| DSA-AI            | Proposed in the 2023 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| MTSGQS            | Proposed in the 2023 IEEE-TITS. | [Paper](link)       | [Link](link)       |

### Baseline Strategies:

| Strategy          | Description                                                                                                           | Reference            | Code Source       |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| Random Sampling (RS)  | Random Sampling (RS) serves as a simple baseline for active learning, randomly selecting data samples for labeling without considering their informativeness. | [Paper](link)       | [Link](link)       |
| Uncertainty Sampling with Fixed Threshold (US_fix)  | Uncertainty Sampling with Fixed Threshold (US_fix) selects samples with uncertainty scores exceeding a fixed threshold for labeling, effectively targeting uncertain regions of the data space. | [Paper](link)       | [Link](link)       |
| Uncertainty Sampling with Variable Threshold (US_var)  | Uncertainty Sampling with Variable Threshold (US_var) dynamically adjusts the uncertainty threshold based on model confidence and dataset characteristics, offering improved sample selection flexibility and performance in dynamic environments. | [Paper](link)       | [Link](link)       |

### OSSL Classifiers:

Discover online semi-supervised learning classifiers in the **OSSL_strategies** folder:

| Classifier        | Description                                                                                                           | Reference            | Code Source        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| OSSBLS            |  Proposed in the 2021 IEEE-TII. | [Paper](link)       | [Link](link)       |
| ISSBLS            |  Proposed in the 2021 IEEE-TII. | [Paper](link)       | [Link](link)       |
| SOSELM            | Proposed in the 2016 Neurocomputing. | [Paper](link)       | [Link](link)       |

### Supervised Classifiers:

Find various online learning classifiers in the **classifer** folder:

| Classifier        | Description                                                                                                           | Reference            | Code Source        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| OLI2DS            | Proposed in the 2023 IEEE-TKDE. | [Paper](link)       | [Link](link)       |
| DES               | Proposed in the 2022 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| BLS-W             | Proposed in the 2021 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| ACDWM             | Proposed in the 2020 IEEE-TNNLS. | [Paper](link)       | [Link](link)       |
| SRP               | Proposed in the 2019 ICDM. | [Paper](link)       | [Link](link)       |
| ARF               | Proposed in the 2017 Machine Learning. | [Paper](link)       | [Link](link)       |


### Datasets:

The **datasets** folder contains .csv files structured with attributes, headers, and labels, catering to the needs of various strategies and classifiers.

### Implementation:

The specific implementations are encapsulated into a unified form. Further technical details and improvements can be explored within each strategy or classifier.

## Environment Setup:

To set up the required environment, run the following command:

```
conda env create -f env.yml
```

[//]: # (## Prerequisites)

[//]: # ()
[//]: # (| Package                    | Version    | Package                    | Version          | Package           | Version          |)

[//]: # (|-------------------------|------------|-------------------------|-----------------|-------------------|-----------------|)

[//]: # (| brotlipy                | 0.7.0      | m2w64-gmp               | 6.1.0           | python            | 3.7.1           |)

[//]: # (| ca-certificates         | 2023.08.22 | m2w64-libwinpthread-git | 5.0.0.4634.697f757 | python-dateutil   | 2.8.2           |)

[//]: # (| certifi                 | 2022.12.7  | matplotlib-base         | 3.2.2           | python_abi        | 3.7             |)

[//]: # (| cffi                    | 1.15.1     | mkl                     | 2020.4          | pytz              | 2023.3.post1    |)

[//]: # (| charset-normalizer      | 2.0.4      | msys2-conda-epoch       | 20160418        | qdldl-python      | 0.1.5           |)

[//]: # (| cryptography            | 39.0.1     | numpy                   | 1.21.6          | requests          | 2.28.1          |)

[//]: # (| cvxpy                   | 1.2.1      | openssl                 | 1.1.1w          | scikit-learn      | 0.22.1          |)

[//]: # (| cvxpy-base              | 1.2.1      | osqp                    | 0.6.2.post0     | scikit-multiflow  | 0.5.3           |)

[//]: # (| cycler                  | 0.11.0     | pandas                  | 1.2.3           | scipy             | 1.7.3           |)

[//]: # (| ecos                    | 2.0.10     | pip                     | 22.3.1          | scs               | 3.2.0           |)

[//]: # (| flit-core               | 3.6.0      | prettytable             | 3.5.0           | setuptools        | 59.8.0          |)

[//]: # (| freetype                | 2.10.4     | pycparser               | 2.21            | six               | 1.16.0          |)

[//]: # (| idna                    | 3.4        | pyopenssl               | 23.0.0          | sortedcontainers  | 2.4.0           |)

[//]: # (| importlib-metadata      | 4.11.3     | pyparsing               | 3.1.1           | tornado           | 6.2             |)

[//]: # (| intel-openmp            | 2023.2.0   | pysocks                 | 1.7.1           | typing_extensions | 4.4.0           |)

[//]: # (| joblib                  | 1.3.2      | python                  | 3.7.1           | urllib3           | 1.26.14         |)

[//]: # (| kiwisolver              | 1.4.4      | python-dateutil         | 2.8.2           | vc                | 14.2            |)

[//]: # (| libblas                 | 3.9.0      | python_abi              | 3.7             | vs2015_runtime    | 14.27.29016     |)

[//]: # (| libcblas                | 3.9.0      | pytz                    | 2023.3.post1    | wcwidth           | 0.2.5           |)

[//]: # (| liblapack               | 3.9.0      | qdldl-python            | 0.1.5           | wheel             | 0.38.4          |)

[//]: # (| libpng                  | 1.6.39     | requests                | 2.28.1          | win_inet_pton     | 1.1.0           |)

## References:

Explore related resources and inspiration at:

- [GitHub - deep-active-learning](https://github.com/ej0cl6/deep-active-learning)

---

## Note

We hope this toolkit serves as a valuable asset in your online learning endeavors.

We are from the THUFDD Research Group, led by Prof. Xiao He and Prof. Donghua Zhou in the Department of Automation at Tsinghua University.

We are committed to the long-term development of this toolkit. Should you have any inquiries, feedback, or wish to contribute, we encourage you to engage with our community via [mailto:liuzy21@mails.tsinghua.edu.cn]. Here's to a fruitful learning journey!

