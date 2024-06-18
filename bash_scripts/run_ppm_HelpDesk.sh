export DATASET=HelpDesk
export MODEL=dalstm
export SPLIT_MODE=holdout
export SEED=42
export EXP_DIR=./results
export N_STEPS=1000
export RUN_NAME=prior_dalstm
export LOSS=card_conditional
export TASK=dalstm_CARD_HelpDesk
export N_SPLITS=1
export N_THREADS=4
export DEVICE_ID=0
export CAT_F_PHI=_cat_f_phi
export MODEL_VERSION_DIR=card_conditional_ppm_results/${N_STEPS}steps/dalstm/${RUN_NAME}/f_phi_prior${CAT_F_PHI}

#python preprocess.py --datasets ${DATASET} --normalization
#python main.py --dataset ${DATASET} --model ${MODEL} --UQ deterministic --split_mode ${SPLIT_MODE} --seed ${SEED} --device ${DEVICE_ID}
#python main.py --dataset ${DATASET} --model ${MODEL} --UQ DA --split_mode ${SPLIT_MODE} --seed ${SEED} --device ${DEVICE_ID}
#python main.py --dataset ${DATASET} --model ${MODEL} --UQ CDA --split_mode ${SPLIT_MODE} --seed ${SEED} --device ${DEVICE_ID}
#python main.py --dataset ${DATASET} --model ${MODEL} --UQ DA_A --split_mode ${SPLIT_MODE} --seed ${SEED} --device ${DEVICE_ID}
#python main.py --dataset ${DATASET} --model ${MODEL} --UQ CDA_A --split_mode ${SPLIT_MODE} --seed ${SEED} --device ${DEVICE_ID}
python main.py --UQ CARD --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config cfg/${TASK}.yml --seed ${SEED} #--train_guidance_only
python main.py --UQ CARD --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test --seed ${SEED}
#python main.py --UQ CARD --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --noise_prior --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config cfg/${TASK}.yml #--train_guidance_only
#python main.py --UQ CARD --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --noise_prior --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test
