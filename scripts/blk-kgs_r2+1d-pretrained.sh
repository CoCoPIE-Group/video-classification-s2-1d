CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --multi-gpu \
	--arch r2+1d-pretrained --dataset ucf101 \
	--sparsity-type blk-kgs --config-file r2+1d-pretrained_2.56x --connectivity-block-size 8 4 \
	--admm --rho 0.0001 --rho-num 4 \
	--epoch 50 --lr 5e-4 --optmzr sgd \
	--log-interval 50 \
	--smooth --smooth-eps 0.1 &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --multi-gpu \
	--arch r2+1d-pretrained --dataset ucf101 \
	--sparsity-type blk-kgs --config-file r2+1d-pretrained_2.56x --connectivity-block-size 8 4 \
	--combine-progressive --masked-retrain --rho 0.0001 --rho-num 4 \
	--epoch 130 --lr 5e-4 --optmzr sgd \
	--log-interval 50 \
	--warmup --warmup-lr 1e-5 --lr-scheduler cosine \
	--distill --teacharch r2+1d-pretrained --teacher-path checkpoint/ucf101_r2+1d-pretrained_transfer_epoch-14_top1-93.981.pt #--smooth --smooth-eps 0.1