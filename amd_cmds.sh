export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OMP_PROC_BIND=true
export OMP_PLACES=cores

python3 train_cortex_phase2_tinyimagenet.py \
  --tiny_root /data/tiny-imagenet-200 \
  --steps 12000 \
  --batch_real 64 \
  --batch_synth 32 \
  --lambda_real 1.0 \
  --save brain_vector_v12.pth