# ExplainableEcgClassification

Welcome to the **ExplainableEcgClassification** project, an advanced research-driven approach for ECG-based arrhythmia detection using deep learning, with a focus on enhancing transparency and interpretability of model decisions. This repository accompanies the manuscript titled **"Towards Transparent AI in Medicine: ECG-Based Arrhythmia Detection with Explainable Deep Learning,"** authored by Oleksii Kovalchuk, Oleksandr Barmak, Pavlo Radiuk, Liliana Klymenko, and Iurii Krak.

> **Preprint**: The preprint version of the paper is available on Preprints.org: [https://www.preprints.org/manuscript/202411.0931/v1](https://www.preprints.org/manuscript/202411.0931/v1)

## Overview

This repository implements a novel approach to arrhythmia detection in ECG signals, leveraging explainable artificial intelligence (XAI) to make model outputs more interpretable for medical professionals. The core contribution is a pipeline integrating:
- **Enhanced R Peak Detection**: Incorporates domain knowledge to increase the precision of R peak identification.
- **Arrhythmia Classification with Deep Learning**: A modified CNN architecture that processes triads of ECG cycles, enabling the model to capture temporal dependencies critical to accurate arrhythmia detection.
- **Clinical Interpretation of Model Decisions**: Translates CNN classification outputs into clinically relevant features, facilitating decision-making by clinicians.

This approach demonstrates high accuracy, achieving near-perfect performance on critical arrhythmia classes, and is designed to bridge the gap between deep learning models and real-world medical usability.

## Features

- **Explainability and Transparency**: Enhances the transparency of model decisions, presenting results in ways understandable to medical experts.
- **High Performance**: Achieves state-of-the-art accuracy on the MIT-BIH and other standard ECG datasets.
- **Adaptability**: The model and code are adaptable for various ECG datasets, and extendable to different arrhythmia classification tasks.
  
## Contents

- **Notebooks/**: Contains Jupyter notebooks for data exploration, model training, and evaluation.
- **README.md**: This file, describing the project structure, installation, and usage.
- **LICENSE**: Licensing information.

## Citation

If you find this project helpful for your research or applications, please consider citing our paper:

```bibtex
@article{kovalchuk2024ecgclassification,
  title={Towards Transparent AI in Medicine: ECG-Based Arrhythmia Detection with Explainable Deep Learning},
  author={Kovalchuk, Oleksii and Barmak, Oleksandr and Radiuk, Pavlo and Klymenko, Liliana and Krak, Iurii},
  journal={Preprints},
  year={2024},
  url={https://www.preprints.org/manuscript/202411.0931/v1}
}
```

## Installation

To set up the project, clone the repository and install the necessary Python libraries:

```bash
git clone https://github.com/okovalchuk98/ExplainableEcgClassification.git
cd ExplainableEcgClassification
pip install -r requirements.txt
```

Ensure you have a compatible version of Python (>=3.8).

## Usage

### 1. Data Preparation
This project uses standard ECG datasets such as **MIT-BIH Arrhythmia Database**. Before training, ensure your data is preprocessed and segmented to fit the input requirements specified in the notebooks.

### 2. Model Training
Navigate to the `notebooks/` directory and execute the Jupyter notebooks to:
- Train the CNN model for R peak detection.
- Classify arrhythmias based on the ECG cycles.
- Interpret and visualize model outputs in a clinically meaningful way.

### 3. Evaluation and Interpretation
Evaluate model performance using classification metrics and visualize ECG signal features that support interpretability for clinicians.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License

This project is licensed under the terms of the The MIT License. See the LICENSE file for more details.

## Acknowledgments

This work was developed as part of a research project at Khmelnytskyi National University, in collaboration with Shupyk National Healthcare University and Taras Shevchenko National University of Kyiv. Special thanks to all authors and contributors for their efforts and insights into making this work clinically applicable and impactful.

For any questions, please contact the corresponding author, Pavlo Radiuk, at [radiukp@khmnu.edu.ua](mailto:radiukp@khmnu.edu.ua).
