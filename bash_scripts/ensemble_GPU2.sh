export MODEL=dalstm
export SPLIT_MODE=holdout
export N_SPLITS=1
export SEEDS=42
export SEED=42
export N_THREADS=4
export DEVICE_ID=7


export DATASET=BPIC12

python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_b --num_models 20
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_b_mve --num_models 20
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_t --num_models 5
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_t_mve --num_models 5

export DATASET=BPIC13I

python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_b --num_models 20
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_b_mve --num_models 20
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_t --num_models 5
python main.py --dataset ${DATASET} --model ${MODEL} --split_mode ${SPLIT_MODE} --seed ${SEEDS} --device ${DEVICE_ID} --UQ en_t_mve --num_models 5

