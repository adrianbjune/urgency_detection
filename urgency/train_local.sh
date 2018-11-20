MODEL_DIR=${PWD} # loc of module and scripts

echo "Deleting old checkpoints..."
rm -r ${MODEL_DIR}/checkpoints

gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${MODEL_DIR}/trainer \
    --train_data_path=gs://urgency-detection/labelled_train.csv
    --eval_data_path=gs://urgency-detection/labelled_test.csv