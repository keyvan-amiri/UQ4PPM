export MODEL=dalstm
export N_SPLITS=1
export SEEDS=42
export N_THREADS=4
export DEVICE_ID=0

export DATASET=BPIC20DD
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}


export DATASET=BPIC20ID
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}


export DATASET=BPIC20RFP
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}


export DATASET=BPIC20PTC
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}


export DATASET=BPIC20TPD
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}


export DATASET=BPIC15_1
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}

export DATASET=BPIC13I
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}

export DATASET=BPIC12
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}

export DATASET=HelpDesk
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}

export DATASET=Sepsis
export CFG=RPA.yaml
python RPA.py --dataset ${DATASET} --cfg ${CFG}
python comparison.py --dataset ${DATASET} --model ${MODEL}