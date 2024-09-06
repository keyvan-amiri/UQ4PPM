export MODEL=dalstm
export SPLIT_MODE=holdout
export N_SPLITS=1
export SEEDS=42
export SEED=42
export N_THREADS=4
export DEVICE_ID=7



export DATASET=BPIC12

python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID} --UQ CARD --test 


