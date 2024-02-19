export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# ReferItGame
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 24 --use_mae --aug_crop --aug_scale --aug_translate --dataset unc --max_query_len 20 --output_dir outputs/refcoco_20x20_vl512 --lr_scheduler poly --epoch 20 --imsize 320
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 24 --use_mae --aug_crop --aug_scale --aug_translate --dataset unc --max_query_len 20 --output_dir outputs/refcoco_20x20_vl512 --epoch 40 --imsize 320 --finetune outputs/refcoco_20x20_vl512/checkpoint.pth --lr 1e-5 --lr_bert 1e-6 --lr_visu 1e-6
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 24 --use_mae --aug_crop --aug_scale --aug_translate --dataset unc --max_query_len 20 --output_dir outputs/refcoco_20x20_vl512 --epoch 60 --imsize 320 --finetune outputs/refcoco_20x20_vl512/checkpoint.pth --lr 1e-5 --lr_bert 1e-6 --lr_visu 1e-6


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 24 --use_mae --aug_crop --aug_scale --aug_translate --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco+_20x20_vl512 --lr_scheduler poly --epoch 30 --imsize 320
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 24 --use_mae --aug_crop --aug_scale --aug_translate --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco+_20x20_vl512 --epoch 60 --imsize 320 --finetune outputs/refcoco+_20x20_vl512/checkpoint.pth --lr 1e-5 --lr_bert 1e-6 --lr_visu 1e-6
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 24 --use_mae --aug_crop --aug_scale --aug_translate --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco+_20x20_vl512 --epoch 90 --imsize 320 --finetune outputs/refcoco+_20x20_vl512/checkpoint.pth --lr 1e-5 --lr_bert 1e-6 --lr_visu 1e-6


# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --use_mae --aug_crop --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_20x20_vl512 --lr_scheduler poly --epoch 20 --imsize 320
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --use_mae --aug_crop --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_20x20_vl512 --epoch 40 --imsize 320 --finetune outputs/refcocog_20x20_vl512/checkpoint.pth --lr 1e-5 --lr_bert 1e-6 --lr_visu 1e-6
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --use_mae --aug_crop --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_20x20_vl512 --epoch 60 --imsize 320 --finetune outputs/refcocog_20x20_vl512/checkpoint.pth --lr 1e-5 --lr_bert 1e-6 --lr_visu 1e-6

# debug
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 11 --use_mae --aug_crop --aug_scale --aug_translate --dataset unc --max_query_len 20 --output_dir outputs/refcoco_test --epoch 20 --imsize 320

