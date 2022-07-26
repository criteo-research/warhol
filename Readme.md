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

python train_warhol.py --image_text_folder /home/u.tanielian/fashion_kaggle/ --vqgan_model_path /home/u.tanielian/vqgan_imagenet_f16_16384_model.ckpt --vqgan_config_path /home/u.tanielian/vqgan_imagenet_f16_16384_configs.yaml --taming --keep_n_checkpoints 4 --truncate_captions --inferring_clip_embeddings --use_of_clip_embed both

### With WDS
python train_warhol.py --image_text_folder "/home/u.tanielian/training_warhol_US_shoes_150k" --vqgan_model_path /home/u.tanielian/vqgan_imagenet_f16_16384.ckpt --vqgan_config_path /home/u.tanielian/vqgan_imagenet_f16_16384_configs.yaml --taming --truncate_captions --wds jpg,txt --inferring_clip_embeddings --keep_n_checkpoints 3 --learning_rate 1e-4 --batch_size 8 --depth 16

python train_warhol_txtClip.py --image_text_folder "/home/u.tanielian/training_warhol_US_shoes_150k" --vqgan_model_path /home/u.tanielian/vqgan_imagenet_f16_16384.ckpt --vqgan_config_path /home/u.tanielian/vqgan_imagenet_f16_16384_configs.yaml --taming --truncate_captions --wds jpg,txt --inferring_clip_embeddings --keep_n_checkpoints 1 --learning_rate 1e-4 --batch_size 8 --depth 32

For a pre-trained one:
python train_warhol.py --image_text_folder "/home/u.tanielian/fashion_kaggle" --vqgan_model_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384.ckpt --vqgan_config_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384_configs.yaml --taming --keep_n_checkpoints 4 --truncate_captions --warhol_path ./wandb/run-20211216_180248-1120qj97/files/warhol.pt --epochs 50 --keep_n_checkpoints 2 --learning_rate 1e-4

python train_warhol.py --image_text_folder "/home/u.tanielian/training_warhol_US_1M" --vqgan_model_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384.ckpt --vqgan_config_path /home/u.tanielian/sync/product_gen/multi-modal-training/taming_models/vqgan_imagenet_f16_16384_configs.yaml --taming --truncate_captions --wds jpg,txt --inferring_clip_embeddings --warhol_path ./wandb/latest-run/files/warhol.pt --keep_n_checkpoints 2 --learning_rate 1e-4 --batch_size 8 --depth 8

python train_warhol.py --image_text_folder "/home/u.tanielian/training_warhol_US_shoes_150k" --vqgan_model_path /home/u.tanielian/vqgan_imagenet_f16_16384.ckpt --vqgan_config_path /home/u.tanielian/vqgan_imagenet_f16_16384_configs.yaml --taming --truncate_captions --wds jpg,txt --inferring_clip_embeddings --keep_n_checkpoints 3 --learning_rate 1e-4 --batch_size 8 --depth 16 --warhol_path ./wandb/run-shoes-150k/files/warhol.pt


