export DEVICE_ID=0

export CFG=dalstm_BPIC20I_card.yml
export CSV=CARD_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID} 


