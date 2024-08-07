export MODEL=dalstm
export SPLIT_MODE=holdout
export N_SPLITS=1
export SEEDS=42
export SEED=42
export N_THREADS=4
export DEVICE_ID=7


export DATASET=HelpDesk

python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID}  --UQ CARD 
python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID} --UQ CARD --test 
#python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID}  --UQ CARD --noise_prior
#python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID} --UQ CARD --noise_prior --test 


export DATASET=Sepsis

python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID}  --UQ CARD 
python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID} --UQ CARD --test 


export DATASET=BPIC20I

python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID}  --UQ CARD --resume_training
python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID} --UQ CARD --test 

export DATASET=BPIC12

python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID}  --UQ CARD
python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID} --UQ CARD --test 

export DATASET=BPIC13I

python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID}  --UQ CARD
python main.py --dataset ${DATASET} --model ${MODEL} --n_splits ${N_SPLITS} --seed ${SEED} --device ${DEVICE_ID} --UQ CARD --test 


