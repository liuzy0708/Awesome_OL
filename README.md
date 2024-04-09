## Description

- **utils.py** is a component file and is the interface between classifiers, and strategies;


- The **OAL_strategies** folder contains the implemented *online active learning* strategies, including:
   - Recent Advanced Progress:
     - ROALE-DI: 2023 IEEE-TKDE
     - CogDQS: 2023 IEEE-TNNLS
     - DSA-AI: 2023 IEEE-TNNLS
     - MTSGQS: 2023 IEEE-TITS
   - And 3 baseline strategies:
     - Random Sampling (RS)
     - Uncertainty Sampling with Fixed threshold (US_fix)
     - Uncertainty Sampling with Variable threshold (US_var)
   

    
- The **OSSL_strategies** folder contains the implemented *online semi-supervised learning* classifiers, including:
   - Recent Advanced Progress:
     - OSSBLS: 2021 IEEE-TII
     - ISSBLS: 2021 IEEE-TII
     - SOSELM: 2016 Neurocomputing


- The **classifer** folder contains the implemented *online learning* classifiers, including:
  - Recent Advanced Progress:
    - OLI2DS: 2023 IEEE-TKDE
    - DES: 2022 IEEE-TNNLS
    - BLS-W: 2021 IEEE-TNNLS
    - ACDWM: 2020 IEEE-TNNLS
    - SRP: 2019 ICDM
    - ARF: 2017 Machine Learning
     
 
- The **datasets** folder contains .csv files. The first line of the .csv file is Header, the last line is labels, and the previous lines are attributes. The strategy is defined in class form rather than function form because the class form can store some changes in data volume, while functions are one-time.


- The specific implementation has been encapsulated into a unified form, and other technical details can be improved in the strategy.


## Environment
- conda env create -f env.yml


## Prerequisites

| Package                    | Version    | Package                    | Version          | Package           | Version          |
|-------------------------|------------|-------------------------|-----------------|-------------------|-----------------|
| brotlipy                | 0.7.0      | m2w64-gmp               | 6.1.0           | python            | 3.7.1           |
| ca-certificates         | 2023.08.22 | m2w64-libwinpthread-git | 5.0.0.4634.697f757 | python-dateutil   | 2.8.2           |
| certifi                 | 2022.12.7  | matplotlib-base         | 3.2.2           | python_abi        | 3.7             |
| cffi                    | 1.15.1     | mkl                     | 2020.4          | pytz              | 2023.3.post1    |
| charset-normalizer      | 2.0.4      | msys2-conda-epoch       | 20160418        | qdldl-python      | 0.1.5           |
| cryptography            | 39.0.1     | numpy                   | 1.21.6          | requests          | 2.28.1          |
| cvxpy                   | 1.2.1      | openssl                 | 1.1.1w          | scikit-learn      | 0.22.1          |
| cvxpy-base              | 1.2.1      | osqp                    | 0.6.2.post0     | scikit-multiflow  | 0.5.3           |
| cycler                  | 0.11.0     | pandas                  | 1.2.3           | scipy             | 1.7.3           |
| ecos                    | 2.0.10     | pip                     | 22.3.1          | scs               | 3.2.0           |
| flit-core               | 3.6.0      | prettytable             | 3.5.0           | setuptools        | 59.8.0          |
| freetype                | 2.10.4     | pycparser               | 2.21            | six               | 1.16.0          |
| idna                    | 3.4        | pyopenssl               | 23.0.0          | sortedcontainers  | 2.4.0           |
| importlib-metadata      | 4.11.3     | pyparsing               | 3.1.1           | tornado           | 6.2             |
| intel-openmp            | 2023.2.0   | pysocks                 | 1.7.1           | typing_extensions | 4.4.0           |
| joblib                  | 1.3.2      | python                  | 3.7.1           | urllib3           | 1.26.14         |
| kiwisolver              | 1.4.4      | python-dateutil         | 2.8.2           | vc                | 14.2            |
| libblas                 | 3.9.0      | python_abi              | 3.7             | vs2015_runtime    | 14.27.29016     |
| libcblas                | 3.9.0      | pytz                    | 2023.3.post1    | wcwidth           | 0.2.5           |
| liblapack               | 3.9.0      | qdldl-python            | 0.1.5           | wheel             | 0.38.4          |
| libpng                  | 1.6.39     | requests                | 2.28.1          | win_inet_pton     | 1.1.0           |


## References
- https://github.com/ej0cl6/deep-active-learning
