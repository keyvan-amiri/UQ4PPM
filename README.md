# A Simple and Calibrated Approach for Uncertainty-Aware Remaining Time Prediction
This is the supplementary githob repository of the paper: "A Simple and Calibrated Approach for Uncertainty-Aware Remaining Time Prediction".

## Installation

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/keyvan-amiri/UQ4PPM
```

To install and set up the required environment on a Linux system, run the following commands:

```bash
conda create -n horoscope python=3.10
conda activate horoscope
pip install -r requirements.txt
conda clean --all
```

### Feature Extraction
In order to train the deterministic backbone model, or any probabilistic model in the uncertainty module, we need to first transform event data into feature vectors. In principle, our approach supports any kind of neural networks as the backbone model. However, our expriments are based on  
[Data aware LSTM approach](https://ieeexplore.ieee.org/abstract/document/8285184) . Feature extraction for all event logs included in our experiments can be achieved by executing the following script:

```
python preprocess.py --datasets BPIC20DD BPIC20ID BPIC20PTC BPIC20RFP BPIC20TPD BPIC15_1 BPIC13I BPIC12 HelpDesk Sepsis   --model dalstm
```

#### Training and evaluation for a probabilistic model
In order to train and evaluate a probabilistic model, and replicate our experiments, the following script should be executed:

```
bash bash_scripts/Train_eval_calib.sh
```

This bash script, include all execution commands for training and evaluation of all UQ techniques included in our experiments. It also automatically applies calirated regression (CR) on top of the probabilistic model. The bash script, include several commands for executing the main script: **main.py**. The main script accept the following structrure for configuration options:
```
python main.py --dataset BPIC20DD --model dalstm --UQ deterministic --cfg dalstm_deterministic.yaml --seed 42 --device 1
```
More precisely, the configuration options which should be specified by the user aligns with the followin structure:

**dataset** specifies the event log that training and evaluation is conducted for it (e.g., BPIC20DD).

**model** specifies the architecture of the backbone model (i.e., dalstm).

**UQ** specifies the uncertainty quantification (UQ) technique. For instance, in the above example, the deterministic backbone is trained which is a mandatory step for Laplace approximation (LA). User can easily decide on the UQ technqiue by changing the value for this option. For instance, to train and evaluate Laplace approximation the following example should be changed as per follows:

```
python main.py --dataset BPIC20DD --model dalstm --UQ LA --cfg dalstm_LA.yaml --seed 42 --device 1
```

**cfg** specifies the configuration file that is used to save all important parameters with respect to training and evaluation of the model.

**n_splits** this configuration is only applicable to CARD (diffusion based UQ technique). Note that in all of our experiments, we used n_splits=1 which is equivalent to holdout data split. Any value larger than one specifies the number of folds in cross-fold validation data split. 

**seed** specifies the random seed that is used. With the exception of ensembles which requires multiple seed, our expriments are based on the single seed=42.

**device** specifies the machine (GPU) that is used for training and evaluation.
