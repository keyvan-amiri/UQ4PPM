export MODEL=dalstm
export N_SPLITS=1
export SEEDS=42
export N_THREADS=4
export DEVICE_ID=2


export DATASET=HelpDesk
export CFG=dalstm_dropout.yaml

export UQ=DA
#python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}

export CFG=dalstm_HelpDesk_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}

