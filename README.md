# Data-Driven Simulator for Mechanical Circulatory Support with Domain Adversarial Neural Process

This repository contains code for the Domain Adversarial Neural Process (DANP) and various baselines reported in the paper, *Data-Driven Simulator for Mechanical Circulatory Support with Domain Adversarial Neural Process*. The data used in this project cannot be shared publicly due to privacy and confidentiality agreements. However, we provide the necessary scripts and instructions to run the model with your own data.


## | Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Setup [wandb](https://wandb.ai/site/) for model evaluation and dashboard. 

## | Usage 

### Preparing Your Data

Since the original data used in the paper cannot be shared, you will need to prepare your own dataset. Ensure your data is in the appropriate format as described in the `data_preparation.md` file.


### Running the Model

Execute the training script with your dataset.

```bash
python src/model/DANP.py -cohort cohort_name -seq_steps 90 -cuda cuda_number -nepochs 150 -lr 0.001 -bc 64
```