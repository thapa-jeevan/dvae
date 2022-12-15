# Face Generation using Discrete VAE

Please, follow the following steps for model training and evaluation.

### Training VAE model

```shell
python train_dvae.py --img_folder data/CelebAMask-HQ/CelebA-HQ-img --batch_size 64
```

### Generate results for transformer training
```shell
python generate_dvae_embeddings.py --img_folder data/CelebAMask-HQ/CelebA-HQ-img --batch_size 128 --img_size 128
```

### Training the transformer
```shell
python train_transformer.py --learning_rate 0.0003 --batch_size 32  --array_folder data/dvae_codes/
```

## Generate Images
```shell
python generate_images.py --vae_checkpoint_path checkpoints/vae_ckpt.pth --transformer_checkpoint_path checkpoints/transformer_ckpt.pt
```
The generated images are placed in `results/transformer_training` folder.