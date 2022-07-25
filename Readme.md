## ProductGen ! 
This is the official repo for Warhol. 

## Setting up the environment ! 
Run install_warhol.sh

## How to run the notebook ?
Run it with the Warhol just created.

## How to launch a training 
python train_warhol.py \
--image_text_folder "pipe:hdfs dfs -cat /user/g.lagarde/webdatasets/US/{00000..29999}.tar" \
--vqgan_model_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384.ckpt \
--vqgan_config_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384_configs.yaml \
--inferring_clip_embeddings --taming --keep_n_checkpoints 4 --truncate_captions --wds jpg,txt 

python train_warhol.py \
--image_text_folder "/home/u.tanielian/fashion_kaggle" \
--vqgan_model_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384.ckpt \
--vqgan_config_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384_configs.yaml \
 --taming --keep_n_checkpoints 4 --truncate_captions