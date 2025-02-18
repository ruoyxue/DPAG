DOWNLOAD_DATA_PATH="./LipReading_dataset/original_dataset/lrs2"
TARGET_DATA_PATH="./LipReading_dataset/processed_face/lrs2-re176"
DATASET_NAME="lrs2"

python "prepare_filescp.py" --src $DOWNLOAD_DATA_PATH --dst $TARGET_DATA_PATH --dataset $DATASET_NAME

CUDA_VISIBLE_DEVICES="0,1,2,3" python detect_landmark_list.py --list $DATASET_NAME.scp \
    --shard 20 --process_per_gpu 4 --region face --batch_size 64 --num_workers 1
