CUDA_VISIBLE_DEVICES=2 python train_eval.py \
  --root_dir logs \
  --experiment_name latent_sac \
  --gin_file params.gin \
  --gin_param load_carla_env.port=3000
