export MODEL=dalstm
export N_SPLITS=1
export SEEDS=42
export N_THREADS=4
export DEVICE_ID=1


export DATASET=HelpDesk

export CFG=dalstm_dropout.yaml
export UQ=DA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble.yaml
export UQ=en_t
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_t_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_RF.yaml
export UQ=RF
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=Sepsis

export CFG=dalstm_dropout.yaml
export UQ=DA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble.yaml
export UQ=en_t
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_t_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_RF.yaml
export UQ=RF
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}


export DATASET=BPIC20PTC

export CFG=dalstm_dropout.yaml
export UQ=DA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble.yaml
export UQ=en_t
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_t_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_RF.yaml
export UQ=RF
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}



export DATASET=BPIC20RFP

export CFG=dalstm_dropout.yaml
export UQ=DA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble.yaml
export UQ=en_t
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_t_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_RF.yaml
export UQ=RF
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}


export DATASET=BPIC20DD

export CFG=dalstm_dropout.yaml
export UQ=DA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble.yaml
export UQ=en_t
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_t_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_RF.yaml
export UQ=RF
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}



export DATASET=BPIC20TPD

export CFG=dalstm_dropout.yaml
export UQ=DA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble.yaml
export UQ=en_t
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_t_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_RF.yaml
export UQ=RF
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}


export DATASET=BPIC20ID

export CFG=dalstm_dropout.yaml
export UQ=DA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=DA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA_A
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble.yaml
export UQ=en_t
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_t_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=en_b_mve
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_RF.yaml
export UQ=RF
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}

