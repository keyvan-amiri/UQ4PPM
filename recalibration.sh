export MODEL=dalstm
export SPLIT_MODE=holdout
export N_SPLITS=1
export SEEDS=42
export SEED=42
export N_THREADS=4
export DEVICE_ID=3

export DATASET=HelpDesk
python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=Sepsis 
python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=BPIC12
python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=BPIC13I
python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=BPIC20I
python evaluation.py --dataset ${DATASET} --model ${MODEL} 




