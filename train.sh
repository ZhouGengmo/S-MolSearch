#!/bin/bash

# Model and Training Parameters
n_gpu=$(nvidia-smi -L | wc -l)
exp_name=${EXP_NAME:-'train'}
clip_dim=768
update_freq=1
tem_soft=1.0
epoch=2
period_num=4
lr_shrink=1
tem_clip=0.05
reg_loss=0.1
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
lr=1e-4
wd=1e-4
dropout=0.1
warmup=0.06
local_batch_size=32
global_batch_size=`expr $local_batch_size \* $n_gpu \* $update_freq`
seed=42

# Data and Model Paths Configuration
data_path=${DATA_PATH:-"chembl"}
echo 'Data path:' $data_path
weight_path=${WEIGHT_PATH:-"mol_pre_no_h_220816.pt"}
echo 'Pretrained weight path:' $weight_path
ft_sup_model=$weight_path
ft_unsup_model=$weight_path
unsup_data_path=${UNSUP_DATA_PATH:-"all_data.lmdb"}
echo 'Unsup data path:' $unsup_data_path
updates_per_epoch=148528  # calculate based on unsup dataset size and batch size, adjust based on your environment

# Automatic Configuration
max_update=$((epoch * updates_per_epoch))
warmup_updates=$(echo "$max_update * $warmup" | bc | xargs printf "%1.0f")
cosine_period=$(( (max_update - warmup_updates) / period_num ))
log_dir=${LOG_DIR:-'./logs'}
save_name=${exp_name}_ep_${epoch}_bs_${global_batch_size}

save_dir="${log_dir}/${save_name}"
mkdir -p $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1


# Training Command
echo "Starting S-MolSearch training..."
echo "GPUs: $n_gpu"
echo "Global batch size: $global_batch_size"
echo "Save directory: $save_dir"

torchrun --standalone --nproc_per_node=$n_gpu \
    $(which unicore-train) $data_path \
    --user-dir ./unimol --train-subset train --valid-subset valid --unsup-data-path $unsup_data_path \
    --num-workers 8 --ddp-backend=c10d \
    --task smolsearch --loss smolsearch --arch smolsearch \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-8 --clip-norm 1.0 --weight-decay $wd \
    --lr-scheduler cosine --lr-period-updates $cosine_period --lr $lr --lr-shrink $lr_shrink \
    --warmup-updates $warmup_updates --max-epoch $epoch --max-update $max_update \
    --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $local_batch_size \
    --update-freq $update_freq --seed $seed \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --log-interval 100 --log-format simple \
    --finetune-sup-model $ft_sup_model \
    --finetune-unsup-model $ft_unsup_model \
    --validate-interval-updates 3000 --save-interval-updates 3000 --keep-interval-updates 5 --no-epoch-checkpoints \
    --tem-clip $tem_clip --tem-soft $tem_soft --tem-logit $tem_clip --sup-clip-dim $clip_dim --unsup-clip-dim $clip_dim \
    --save-dir $save_dir --only-polar $only_polar \
    --tensorboard-logdir $save_dir/tsb --soft-loss 1.0 --reg-loss $reg_loss \
    2>&1 | tee $log_dir/${save_name}.log

echo "Training completed. Results saved to: $save_dir"
echo "Log saved to: $log_dir/${save_name}.log"
