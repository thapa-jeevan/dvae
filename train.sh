# VAE training
python train.py --img_folder /home/jt4812/data/CelebAMask-HQ/CelebA-HQ-img --loadVAE /home/jt4812/Projects/dvae/checkpoints/vae-76.pth --batch_size 64

# transformer training
CUDA_VISIBLE_DEVICES=1 python train_transformer.py --pkeep 0.5 --experiment-name lr0.0003 --learning-rate 0.0003 --batch-size 64  --array-folder data/dvae_codes/ --vae-checkpoint-path checkpoints/vae-230.pth --epoch 230 --gen-cuda 2