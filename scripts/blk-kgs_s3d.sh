CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --multi-gpu \
	--arch s3d --dataset ucf101 \
	--sparsity-type blk-kgs --config-file s3d_2.09x --connectivity-block-size 8 4 \
	--admm --rho 0.0001 --rho-num 1 \
	--epoch 1 --lr 5e-4 --optmzr sgd \
	--log-interval 50 \
	--smooth --smooth-eps 0.1 &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --multi-gpu \
	--arch s3d --dataset ucf101 \
	--sparsity-type blk-kgs --config-file s3d_2.09x --connectivity-block-size 8 4 \
	--combine-progressive --masked-retrain --rho 0.0001 --rho-num 1 \
	--epoch 1 --lr 5e-4 --optmzr sgd \
	--log-interval 50 \
	--warmup --warmup-lr 1e-5 --lr-scheduler cosine \
	--distill --teacharch s3d --teacher-path checkpoint/ucf101-s3d-ts-max-f16-multisteps-bs32-e20_epoch-20_top1-90.573.pt #--smooth --smooth-eps 0.1