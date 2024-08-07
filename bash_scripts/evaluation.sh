export MODEL=dalstm

export DATASET=HelpDesk
python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=Sepsis 
python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=BPIC12
#python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=BPIC13I
#python evaluation.py --dataset ${DATASET} --model ${MODEL} 

export DATASET=BPIC20I
python evaluation.py --dataset ${DATASET} --model ${MODEL} 




