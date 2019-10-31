gpuid=$1

export CUDA_VISIBLE_DEVICES=$gpuid
echo $CUDA_VISIBLE_DEVICES

python3 extract_features.py --image_list_file image_list_hpatches_sequences.txt