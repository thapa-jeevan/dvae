# Generate results for transformer training
CUDA_VISIBLE_DEVICES=0 python test.py --img_folder /home/jt4812/data/CelebAMask-HQ/CelebA-HQ-img --loadVAE /home/jt4812/Projects/dvae/checkpoints/128img_0.00007lr-490.pth --batch_size 128 --img_size 128

# Generate Images
CUDA_VISIBLE_DEVICES=0 python generate_images.py --vae-checkpoint-path /home/jt4812/Projects/dvae/checkpoints/128img_0.00007lr-490.pth --transformer-checkpoint-path /home/jt4812/Projects/dvae/checkpoints/transformer_99.pt