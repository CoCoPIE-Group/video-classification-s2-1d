CUDA_VISIBLE_DEVICES=1,2,3 python ./MOpt/rt3d_pruning/opt_main.py --multi-gpu \
	--arch c3d --dataset ucf101 \
	--sparsity-type blk-kgs --config-file c3d_2.53x_v2 --connectivity-block-size 8 4 \
	--admm --rho 0.0001 --rho-num 4 \
	--epoch 1 --lr 5e-4 --optmzr sgd \
	--log-interval 50 \
	--smooth --smooth-eps 0.1 &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./MOpt/rt3d_pruning/opt_main.py --multi-gpu \
	--arch c3d --dataset ucf101 \
	--sparsity-type blk-kgs --config-file c3d_2.53x_v2 --connectivity-block-size 8 4 \
	--combine-progressive --masked-retrain --rho 0.0001 --rho-num 4 \
	--epoch 1 --lr 5e-4 --optmzr sgd \
	--log-interval 50 \
	--warmup --warmup-lr 1e-5 --lr-scheduler cosine \
	--distill --teacharch c3d --teacher-path checkpoint/ucf101_c3d_transfer_epoch-14_top1-81.571.pt #--smooth --smooth-eps 0.1