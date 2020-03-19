CUDA_VISIBLE_DEVICES=0 python train_eval.py \
  --root_dir logs \
  --experiment_name latent_sac_try \
  --gin_file params.gin \
  --gin_param load_carla_env.port=2000
