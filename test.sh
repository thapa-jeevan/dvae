# Generate results for transformer training
python generate_dvae_embeddings.py --img_folder /home/jt4812/data/CelebAMask-HQ/CelebA-HQ-img --vae_checkpoint_path checkpoints/128img_0.00007lr-490.pth --batch_size 128 --img_size 128

# Generate Images
ython generate_images.py --vae_checkpoint_path checkpoints/vae-0.pth --transformer_checkpoint_path checkpoints/lr0.0003_transformer_0.pt