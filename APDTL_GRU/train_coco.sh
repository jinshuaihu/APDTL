DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
  --logger_name runs/${DATASET_NAME}_butd_region_bigru/log --model_name runs/${DATASET_NAME}_butd_region_bigru \
  --num_epochs=25 --batch_size=256 --lr_update=12 --learning_rate=.0008 --precomp_enc_type basic --workers 10 \
  --log_step 400 --embed_size 1024 --vse_mean_warmup_epochs 1
