device=0
name=200
base_dir=${name}
save_dir=./checkpoints/${base_dir}/
mkdir -p ${save_dir}

export CUDA_VISIBLE_DEVICES=${device}
nohup python train.py \
        --dataset visa \
        --train_data_path /data/visa \     # change to your absolute path
        --save_path ${save_dir} \
        --epoch 15 \
        --save_freq 1 \