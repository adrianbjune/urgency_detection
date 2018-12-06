# Parameters
JOB_NAME=urgency_detection_$(date +"%y%m%d_%H%M%S")
SCALE_TIER=BASIC_GPU
BUCKET=gs://urgency-detection
OUTDIR=${BUCKET}/${JOB_NAME}             # training output dir
MODEL_DIR=${PWD} # loc of module and scripts
REGION=us-east1

# Submit job
gcloud ml-engine jobs submit training $JOB_NAME \
    --module-name=trainer.task \
    --package-path=${MODEL_DIR}/trainer \
    --job-dir=$OUTDIR/job \
    --config=${MODEL_DIR}/config.yaml \
    --region=${REGION} \
    -- \
    --output_dir=$OUTDIR/output \
    --train_steps=5000 \
    --train_data_path=labelled_train.csv \
    --eval_data_path=labelled_test.csv \
    --hidden_units="128 32 4"
