# GeometricLens

## Introduction

This Python-based tool is designed for  interpretability research on Large Language Models (LLMs). The workflow handled by this repo conists in retrieving datasets, processing them, and feeding them into a language model to analyze and store the hidden states of instances. Its core functionality consists in integrating  methods from [DADApy](https://dadapy.readthedocs.io/en/latest/) to compute geoemtrical metrics such as *intrinsic dimension*, *neighbour overlap* and *clustering*. 
This repo is still under construction and currently works only on the cluster of my university 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Installation

(Here, provide detailed steps on how to install your software. Include any prerequisites or system requirements.)

```bash
# Example of installation steps
git clone https://github.com/your-repository.git
cd your-repository
pip install -r requirements.txt
```

## Usage

(Describe how to use the program, including basic commands and their explanations.)

```python
# Example of how to run the script
python inference.py --dataset your_dataset.csv
```

## Features

- **Data Handling:** Efficiently retrieves and manipulates datasets for processing.
- **Model Feeding:** Seamlessly feeds processed data into LLMs for analysis.
- **State Analysis:** Captures and stores hidden states of instances from the LLM.
- **Metrics Computation:** Supports two categories of metrics:
  - **Basic NLP Metrics:** Essential metrics for natural language processing analysis.
  - **Geometric Metrics:** Advanced metrics like intrinsic dimension, overlap, and upcoming clustering features.
- **Geometric Analysis:** Utilizes DADapy library for complex computational geometric algorithms.

## Dependencies

- DADApy library
- (List other major dependencies, versions, and any specific requirements.)

## Configuration

(Detail any configuration steps or files required for the program to run.)

## Documentation

(Provide links to any external documentation, wikis, or manuals.)

## Examples

(Share a few examples to demonstrate how your tool can be used in real-world scenarios.)

## Troubleshooting

(Include common issues and their solutions.)

## Contributors

(List the main contributors to this project. You can also include a section on how others can contribute to the project.)

## License

(State the license under which your software is released.)

- [ ] add property
- [ ] add linting
- [ ] add annotations
- [ ] add clustering
- [ ] add slurm runner
