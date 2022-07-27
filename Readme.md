## ProductGen !
This is the official repo for Warhol.

## Setting up the environment !
Follows the instructions in install_warhol.sh

## How to run the notebook ?
Run it with the Warhol venv that has just been created.

## How to launch a training
python train_warhol.py --image_text_folder path_to_folder --vqgan_model_path path_vqgan_model.ckpt --vqgan_config_path path_to_vqgan_config.yaml --taming --keep_n_checkpoints 4 --truncate_captions  --use_of_clip_embed both

### What about the dataset ?
You need a dataset with files
- 0000.jpg
- 0000.txt
- 0000.npy (with both img & txt embeddings)

### What about clip embeddings ?
If you don't have them, use
''' --inferring_clip_embeddings'''
or run the file infer_clip_embeddings:
'''python infer_clip_embeddings.py --inferring_clip_embeddings --image_text_folder path_to_folder --truncate_captions'''
