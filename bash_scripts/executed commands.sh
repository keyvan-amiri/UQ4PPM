export MODEL=dalstm
export SPLIT_MODE=holdout
export N_SPLITS=1
export SEEDS=42
export SEED=42
export N_THREADS=4
export DEVICE_ID=5


export DATASET=HelpDesk

python preprocess.py --datasets ${DATASET} --normalization_lstm False
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ DA 
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ CDA
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ mve


export DATASET=env_permit

python preprocess.py --datasets ${DATASET} --normalization_lstm False
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ DA 
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ CDA
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ mve
