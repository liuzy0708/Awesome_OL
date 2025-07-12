<h1 align="center">
  <img src="https://github.com/liuzy0708/liuzy0708.github.io/blob/main/assets/img/SL.png" width="5%" alt="" />
  Awesome_OL: A General Toolkit for Online Learning Approaches
</h1>

<p align="center">
  <img src="https://img.shields.io/github/stars/liuzy0708/Awesome_OL?style=social" />
  <img src="https://img.shields.io/github/forks/liuzy0708/Awesome_OL?style=social" />
  <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" />
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg" />
  <img src="https://img.shields.io/badge/Anaconda-supported-success.svg" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" />
</p>


<p align="center">
  <img width="80%" alt="Awesome_OL" src="https://github.com/liuzy0708/Awesome_OL/assets/115722686/63b2ae44-b2b4-433d-aafc-db650a46a691" />
</p>

---

## 📖 Table of Contents

- [📖 Table of Contents](#-table-of-contents)
- [🌟 Overview](#-overview)
- [🧠 OAL Strategies](#-oal-strategies)
- [⚙️ OAL Classifiers](#-oal-classifiers)
- [🔍 OSSL Classifiers](#-ossl-classifiers)
- [📊 Supervised Classifiers](#-supervised-classifiers)
- [🧩 Summary of Features](#-summary-of-features)
- [🛠 Usage Guide](#-usage-guide)
  - [🔧 Environment Setup](#-environment-setup)
  - [🧪 Demo](#-demo)
  - [📂 Datasets](#-datasets)
  - [📈 Visualization 📉](#-visualization-)
  - [📜 Logs](#-logs)
  - [🧰 Utility](#-utility)
- [📚 References](#-references)
- [📝 Note](#-note)
- [✨ Contributor Declaration](#-contributor-declaration)
- [👀 Views](#-views)

---

## 🌟 Overview

Welcome to **Awesome_OL**, your comprehensive toolkit for **online learning strategies and classifiers**. This repository includes state-of-the-art implementations for **Online Active Learning (OAL)** and **Online Semi-Supervised Learning (OSSL)**, complete with classifiers, datasets, and visualizations.

For usage instructions, please see the [Usage Guide](#-usage-guide).


---

## 🧠 OAL Strategies

**Explore a variety of online active learning strategies in the `OAL_strategies/` folder.**

| 🧩 Strategy | 📝 Description | 📚 Reference | 💾 Code | 📅 Year | 🏛️ Journal/Conference |
|:-----------|:---------------|:-------------|:--------|:-------:|:-----------------------|
| **CogDQS** | Dual-query strategy using human memory cognition | [IEEE](https://ieeexplore.ieee.org/abstract/document/9492291) | — | **2023** | *TNNLS*                |
| **DSA-AI** | Dynamic submodular learning for imbalanced drifting streams | [IEEE](https://ieeexplore.ieee.org/abstract/document/10195233/) | [GitHub](https://github.com/liuzy0708/DSLS-Demo) | **2024** | *TNNLS*                |
| **MTSGQS** | Memory-triggered submodularity-guided strategy | [IEEE](https://ieeexplore.ieee.org/abstract/document/10105849) | — | **2023** | *TITS*                 |
| **DMI-DD** | Explanation-based query strategy at chunk level | [IEEE](https://ieeexplore.ieee.org/abstract/document/10375819) | [GitHub](https://github.com/liuzy0708/DMI-LS-Demo) | **2024** | *TCYB*                 |

**Baseline Strategies**

| 🧩 Strategy  | 📝 Description                              | 📚 Reference                                                                 | 💾 Code  | 📅 Year | 🏛️ Journal/Conference |
|:------------|:-------------------------------------------|:----------------------------------------------------------------------------|:--------:|:-------:|:---------:|
| **RS**      | Random Sampling                            | —                                                                          | —        | —       | —         |
| **US_fix**  | Uncertainty sampling with fixed threshold | [IEEE](https://ieeexplore.ieee.org/abstract/document/6414645)               | —        | **2014**| *TNNLS*   |
| **US_var**  | Uncertainty sampling with variable threshold | [IEEE](https://ieeexplore.ieee.org/abstract/document/6414645)             | —        | **2014**| *TNNLS*   |

---

## ⚙️ OAL Classifiers

| 🤖 Classifier    | 📝 Description                                  | 📚 Reference                                                               | 💾 Source                                              | 📅 Year | 🏛️ Journal/Conference |
|:----------------|:-----------------------------------------------|:--------------------------------------------------------------------------|:------------------------------------------------------|:-------:|:---------:|
| **ROALE-DI**    | Reinforcement-based ensemble for drifting imbalanced data | [IEEE](https://ieeexplore.ieee.org/abstract/document/9204849)        | [GitHub](https://github.com/saferhand/ROALE-DI)       | **2022**| *TKDE*    |
| **OALE**        | Online ensemble with hybrid labeling            | [IEEE](https://ieeexplore.ieee.org/abstract/document/8401336)        | —                                                     | **2019**| *TNNLS*   |

---

## 🔍 OSSL Classifiers

| 🤖 Classifier  | 📝 Description                                  | 📚 Reference                                                               | 💾 Source | 📅 Year | 🏛️ Journal/Conference |
|:--------------|:-----------------------------------------------|:--------------------------------------------------------------------------|:---------:|:-------:|:---------:|
| **OSSBLS**    | Semi-supervised BLS with static anchors         | [IEEE](https://ieeexplore.ieee.org/abstract/document/9314231)        | —         | **2021**| *TII*    |
| **ISSBLS**    | Semi-supervised BLS without historical dependency| [IEEE](https://ieeexplore.ieee.org/abstract/document/9314231)        | —         | **2021**| *TII*    |

**Baseline Strategy**
| 🤖 Classifier  | 📝 Description          | 📚 Reference                                                                                     | 📅 Year | 🏛️ Journal/Conference  |
|:--------------|:------------------------|:------------------------------------------------------------------------------------------------|:-------:|:--------------:|
| **SOSELM**    | Semi-supervised ELM     | [ScienceDirect Paper](https://www.sciencedirect.com/science/article/pii/S0925231215011212)       | **2016**| *Neurocomputing* |


---

## 📊 Supervised Classifiers

| 🤖 Classifier  | 📝 Description                                       | 📚 Reference                                                                 | 💾 Source                                                           | 📅 Year | 🏛️ Journal/Conference       |
|:--------------|:----------------------------------------------------|:----------------------------------------------------------------------------|:-------------------------------------------------------------------|:-------:|:------------------:|
| **OLI2DS**    | Imbalanced data stream learner with dynamic costs    | [IEEE](https://ieeexplore.ieee.org/abstract/document/10058539)        | [GitHub](https://github.com/youdianlong/OLI2DS)                    | **2023**| *TKDE*             |
| **IWDA**      | Learner-agnostic drift adaptation using density estimation | [IEEE](https://ieeexplore.ieee.org/abstract/document/10105220)    | [GitHub](https://github.com/SirPopiel/IWDA)                        | **2023**| *TNNLS*            |
| **DES**       | Drift-adaptive ensemble with SMOTE                    | [IEEE](https://ieeexplore.ieee.org/abstract/document/9802893)         | [GitHub](https://github.com/Jesen-BT/DES-ICD)                      | **2024**| *TNNLS*            |
| **ACDWM**     | Adaptive chunk selection for stability and drift      | [IEEE](https://ieeexplore.ieee.org/document/8924892)                  | [GitHub](https://github.com/jasonyanglu/ACDWM)                     | **2020**| *TNNLS*            |
| **ARF**       | Adaptive resampling ensemble with ADWIN               | [Springer](https://link.springer.com/article/10.1007/s10994-017-5642-8)| [GitHub](https://github.com/scikit-multiflow/scikit-multiflow)     | **2017**| *Machine Learning*  |
| **SRP**       | Random subspace + online bagging                       | [IEEE](https://ieeexplore.ieee.org/document/8970784)                  | [GitHub](https://github.com/scikit-multiflow/scikit-multiflow)     | **2019**| *ICDM*             |
| **BLS-W**     | Online BLS with Sherman–Morrison–Woodbury update      | [IEEE](https://ieeexplore.ieee.org/abstract/document/10375819)        | [GitHub](https://github.com/liuzy0708/DMI-LS-Demo)                 | **2023**| *TCYB*             |
| **QRBLS**     | BLS with QR factorization                              | [IEEE](https://ieeexplore.ieee.org/abstract/document/4012031)         | [GitHub](https://github.com/Lichen0102/QRBLS)                      | **2025**| *TNNLS*            |

**Baseline Classifier**

| 🤖 Classifier  | 📝 Description                                    | 📚 Reference                                                                 | 💾 Source                                                       | 📅 Year |  🏛️ Journal/Conference   |
|:--------------|:-------------------------------------------------|:----------------------------------------------------------------------------|:---------------------------------------------------------------|:-------:|:-------:|
| **OSELM**     | Sequential ELM without drift detection            | [IEEE](https://ieeexplore.ieee.org/abstract/document/4012031)         | [GitHub](https://github.com/leferrad/pyoselm)                   | **2006**| *TNNLS* |

---

## 🧩 Summary of Features

| 🔹 Method  | 🧠 OAL Strategy | 🤖 Classifier | ⚪ Binary | 🟢 Multi-class | 🔄 Drift Adaptation | 🧩 Ensemble |
|:-----------|:--------------:|:-------------:|:--------:|:--------------:|:-------------------:|:-----------:|
| **ROALE-DI** | ✅             | ✅            | ✅       | ✅             | ✅                  | ✅          |
| **CogDQS**   | ✅             | —             | ✅       | ✅             | ✅                  | —           |
| **DSA-AI**   | ✅             | —             | ✅       | ✅             | ✅                  | —           |
| **DMI-DD**   | ✅             | —             | ✅       | ✅             | ✅                  | —           |
| **MTSGQS**   | ✅             | —             | ✅       | ✅             | ✅                  | —           |
| **RS**       | ✅             | —             | ✅       | ✅             | —                   | —           |
| **US-fix**   | ✅             | —             | ✅       | ✅             | —                   | —           |
| **US-var**   | ✅             | —             | ✅       | ✅             | —                   | —           |
| **OLI2DS**   | —              | ✅            | ✅       | —              | ✅                  | —           |
| **IWDA**     | —              | ✅            | ✅       | ✅             | ✅                  | ✅          |
| **DES**      | —              | ✅            | ✅       | —              | ✅                  | ✅          |
| **ACDWM**    | —              | ✅            | ✅       | —              | ✅                  | ✅          |
| **SRP**      | —              | ✅            | ✅       | ✅             | ✅                  | ✅          |
| **ARF**      | —              | ✅            | ✅       | ✅             | ✅                  | ✅          |
| **QRBLS**    | —              | ✅            | ✅       | ✅             | —                   | —           |



---

## 🛠 Usage Guide

> 👉 This section will guide users on how to use this project. It will  introduce the operation steps including environment preparation, data loading, model selection and visualization results.

---

### 🔧 Environment Setup

💡 Follow these steps to complete the environment setup (using VSCode as an example):

1. **Open Anaconda Prompt or Terminal**  
2. **Navigate to the directory** containing the `env.yml` file  
3. **Create the Conda environment** by running:

    ```bash
    conda env create -f env.yml
    ```

4. **Activate the Conda environment**  
   Open the integrated terminal in VSCode (`Terminal` > `New Terminal`) and execute:

    ```bash
    conda activate OL
    ```

5. **Select Python Interpreter in VSCode**  
   - Press `Ctrl + Shift + P` (Windows/Linux) or `Cmd + Shift + P` (Mac) to open the Command Palette  
   - Type and select **Python: Select Interpreter**  
   - Choose the interpreter corresponding to the activated Conda environment (`OL`)

6. **Run your Python code**  
   Open your Python files and run them as usual. The activated environment will provide all required packages and dependencies.

---

### 🧪 Demo

- In the project root directory, locate the file `main.ipynb`. Within this notebook, you can select the framework, dataset, classifier, strategies, and hyperparameters you wish to use.
- Optionally, you can output visualization results for an intuitive comparison of model performance. These results will also be saved automatically in the `Results` folder.
- For detailed guidance, please follow the step-by-step instructions provided within the notebook.

---

### 📂 Datasets

Datasets are stored as `.csv` files in the `datasets` folder. Each file contains:

🔹 **Attributes** (features)  
🔹 **Labels**

You can select any `.csv` file as your test dataset.

---

### 📈 Visualization 📉

Visualization tools are provided in the `visualization` folder, including:

- Multi-model confusion matrix  
- Dynamic GIFs displaying Accuracy curves and Macro F1 scores

The following example results can be viewed directly in the `main.ipynb`:

<p align="center">
  <img width="80%" src="https://github.com/Alpha0629/Alpha0629.github.io/raw/main/assets/Results_combined_Waveform_all_models.gif" alt="Combined Waveform Animation" />
</p>

<p align="center">
  <img width="80%" src="https://github.com/Alpha0629/Alpha0629.github.io/raw/main/assets/ConfMatrix_Waveform_all_models.png" alt="Confusion Matrix Waveform" />
</p>

---

### 📜 Logs

You can find detailed log information for each demo run in the `Logs` folder located at the project root. Example log snippet:

```txt
16:08 --------------------------------------------------------------------------------------------------
16:08 Max samples: 1000
16:08 n_round: 3
16:08 n_pt: 100
16:08 dataset_name: Waveform
16:08 chunk_size: 1
16:08 framework: OL
16:08 stream: None
16:08 clf_name_list: ['BLS', 'NB', 'ISSBLS', 'OSSBLS', 'DWM']
16:08 num_method: 5
16:08 directory_path: C:\Users\Projects\Online-Learning-Framework\Results\Results_Waveform_OL_100_1_1000
16:08 --------------------------------------------------------------------------------------------------
```

### 🧰 Utility

- `utils.py`: Interfaces between classifiers and strategies, enabling smooth combination and extension.

---

## 📚 References
xplore related resources and inspiration at:

- [GitHub - deep-active-learning](https://github.com/ej0cl6/deep-active-learning)
- [GitHub - scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow)

---

## 📝 Note
We sincerely hope this toolkit becomes a valuable resource in your journey with online learning. Our dedicated team at the **THUFDD Research Group**, led by **Prof. Xiao He** and **Prof. Donghua Zhou** from the Department of Automation at Tsinghua University, is committed to driving innovation and excellence in machine learning applications for industry.

_Wishing you a rewarding and inspiring learning experience!_

Project contributors include:
- Zeyi Liu: 
  - liuzy21@mails.tsinghua.edu.cn
- Songqiao Hu: 
  - hsq23@mails.tsinghua.edu.cn 
- Pengyu Han:
  - hpy24@mails.tsinghua.edu.cn
- Jiaming Liu:
  - 23371007@buaa.edu.cn

--- 

- 🏫 We Are From
<p align="center">
  <img src="https://raw.githubusercontent.com/Alpha0629/Alpha0629.github.io/main/assets/logo.svg" width="200" />
</p>

---

## ✨ Contributor Declaration
If you are interested in becoming a **contributor** to this project, we welcome your participation. Together, we can continue to refine and expand this toolkit to empower researchers, practitioners, and enthusiasts in the field.

Please feel free to get in touch!
- **Contact Person**: Zeyi Liu  
- **Email**: [liuzy21@mails.tsinghua.edu.cn](mailto:liuzy21@mails.tsinghua.edu.cn)

We look forward to your participation and collaboration to push this project forward! 💪

---

## 👀 Views
[![Visitor Map](https://www.clustrmaps.com/map_v2.png?d=LQnBrfl-6T3YvcDDjJwmTwvPbQNLlm52G2eMPCf-LfE&cl=ffffff)](https://clustrmaps.com/site/1c72l)
