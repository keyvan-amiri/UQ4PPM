export MODEL=dalstm
export N_SPLITS=1
export SEEDS=42
export N_THREADS=4
export DEVICE_ID=0

export DATASET=BPIC20DD
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC20DD.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=BPIC20ID
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC20ID.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=BPIC20RFP
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC20RFP.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=BPIC20PTC
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC20PTC.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=BPIC20TPD
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC20TPD.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=BPIC15_1
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC15_1.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=BPIC13I
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC13I.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=BPIC12
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_BPIC12.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=HelpDesk
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_HelpDesk.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}

export DATASET=Sepsis
export CFG=dalstm_deterministic.yaml
export UQ=deterministic
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_mve.yaml
export UQ=H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_LA_Sepsis.yaml
export UQ=LA
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_dropout.yaml
export UQ=DA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export UQ=CDA+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_ensemble_boot.yaml
export UQ=BE+H
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --seed ${SEEDS} --device ${DEVICE_ID}
export CFG=dalstm_CARD.yaml
export UQ=CARD
python main.py --dataset ${DATASET} --model ${MODEL} --UQ ${UQ} --cfg ${CFG} --n_splits ${N_SPLITS} --seed ${SEEDS} --device ${DEVICE_ID}