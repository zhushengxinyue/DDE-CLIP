device=3

export CUDA_VISIBLE_DEVICES=${device}
nohup python test.py \
        --dataset "mvtec" \
        --data_path "./data/mvtec" \  # change to your absolute path
        --checkpoint_path "./checkpoints/test.pth" \
        > "test_output.log" 2>&1 &
            

