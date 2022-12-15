# VAE training
python train_dvae.py --img_folder /home/jt4812/data/CelebAMask-HQ/CelebA-HQ-img --batch_size 64

# transformer training
python train_transformer.py --pkeep 0.5 --experiment_name lr0.0003 --learning_rate 0.0003 --batch_size 32  --array_folder data/dvae_codes/