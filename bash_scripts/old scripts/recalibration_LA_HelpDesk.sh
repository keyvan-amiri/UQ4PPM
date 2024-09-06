export DEVICE_ID=2
export CFG=dalstm_HelpDesk.yaml

export CSV=LA_holdout_seed_42_inference_result_.csv
python recalibration.py --csv_file ${CSV} --cfg_file ${CFG} --device ${DEVICE_ID}