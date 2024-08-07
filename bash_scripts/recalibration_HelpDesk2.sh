export DEVICE_ID=3

export CFG=dalstm_HelpDesk.yaml

export CSV=en_b_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=en_b_mve_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=en_t_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}

export CSV=en_t_mve_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}



