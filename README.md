## Awesome_OL: Online Learning Toolkit

Welcome to Awesome_OL, your comprehensive toolkit for online learning strategies and classifiers! This repository provides a collection of state-of-the-art strategies and classifiers for online active learning (OAL) and online semi-supervised learning (OSSL). Whether you're a researcher, practitioner, or enthusiast in machine learning, this toolkit offers valuable resources and implementations to enhance your projects.

### Contents:

- **utils.py**: This component file serves as the interface between classifiers and strategies, facilitating seamless interaction within the toolkit.

### OAL Strategies:

Explore a variety of online active learning strategies located in the **OAL_strategies** folder:

- **Recent Advanced Progress**:
  - **ROALE-DI**: Introduced in 2023 IEEE-TKDE.
  - **CogDQS**: Presented in 2023 IEEE-TNNLS.
  - **DSA-AI**: Described in 2023 IEEE-TNNLS.
  - **MTSGQS**: Discussed in 2023 IEEE-TITS.
- **Baseline Strategies**:
  - **Random Sampling (RS)**
  - **Uncertainty Sampling with Fixed Threshold (US_fix)**
  - **Uncertainty Sampling with Variable Threshold (US_var)**

### OSSL Classifiers:

Discover online semi-supervised learning classifiers in the **OSSL_strategies** folder:

- **Recent Advanced Progress**:
  - **OSSBLS**: Introduced in 2021 IEEE-TII.
  - **ISSBLS**: Presented in 2021 IEEE-TII.
  - **SOSELM**: Described in 2016 Neurocomputing.

### Online Learning Classifiers:

Find various online learning classifiers in the **classifer** folder:

- **Recent Advanced Progress**:
  - **OLI2DS**: Presented in 2023 IEEE-TKDE.
  - **DES**: Described in 2022 IEEE-TNNLS.
  - **BLS-W**: Introduced in 2021 IEEE-TNNLS.
  - **ACDWM**: Discussed in 2020 IEEE-TNNLS.
  - **SRP**: Presented in 2019 ICDM.
  - **ARF**: Introduced in 2017 Machine Learning.

### Datasets:

The **datasets** folder contains .csv files structured with attributes, headers, and labels, catering to the needs of various strategies and classifiers.

### Implementation:

The specific implementations are encapsulated into a unified form. Further technical details and improvements can be explored within each strategy or classifier.

## Environment Setup:

To set up the required environment, run the following command:

```
conda env create -f env.yml
```

## Prerequisites:

Make sure you have the following packages installed:

| Package                    | Version    | Package                    | Version          | Package           | Version          |
|-------------------------|------------|-------------------------|-----------------|-------------------|-----------------|
| [List of packages]          | [Versions] | [List of packages]          | [Versions]           | [List of packages]           | [Versions]          |

[Update the table with actual package names and versions]

## References:

Explore related resources and inspiration at:

- [GitHub - deep-active-learning](https://github.com/ej0cl6/deep-active-learning)

Thank you for choosing Awesome_OL! We hope this toolkit serves as a valuable asset in your online learning endeavors. If you have any questions, feedback, or contributions, feel free to reach out and join our community. Happy learning!
