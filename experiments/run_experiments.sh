#!/bin/bash
set -e
cd ..
GROUP=maple

# ############# TRAIN #####################
python main_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-0-maple --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --mode run --epochs 50 --batch_size 2 --group $GROUP
python main_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-1-maple --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --mode run --epochs 50 --batch_size 2 --group $GROUP
python main_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-2-maple --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --mode run --epochs 50 --batch_size 2 --group $GROUP
python main_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-3-maple --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --mode run --epochs 50 --batch_size 2 --group $GROUP
python main_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-4-maple --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --mode run --epochs 50 --batch_size 2 --group $GROUP

# ############# EVALUATE #################
python evaluate_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-0-maple --group $GROUP --fold 0 --dataset twitter --window_size 128 --output_dir maple_twitter
python evaluate_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-1-maple --group $GROUP --fold 1 --dataset twitter --window_size 128 --output_dir maple_twitter
python evaluate_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-2-maple --group $GROUP --fold 2 --dataset twitter --window_size 128 --output_dir maple_twitter
python evaluate_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-3-maple --group $GROUP --fold 3 --dataset twitter --window_size 128 --output_dir maple_twitter
python evaluate_maple.py  --config_file configs/combos/clip_clip.yaml --name fold-4-maple --group $GROUP --fold 4 --dataset twitter --window_size 128 --output_dir maple_twitter