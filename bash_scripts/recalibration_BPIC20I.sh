export DEVICE_ID=0

export CFG=dalstm_BPIC20I.yaml

export CSV=CDA_A_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=CDA_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=DA_A_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=DA_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=mve_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID} 

export CSV=en_b_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=en_b_mve_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=en_t_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=en_t_mve_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CFG=dalstm_BPIC20I_card.yml
export CSV=CARD_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID} 


