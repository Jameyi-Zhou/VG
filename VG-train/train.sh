export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ReferItGame
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --lr 5e-6 --lr_bert 5e-7 --lr_visu_tra 5e-7 --lr_visu_cnn 5e-7 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --vl_enc_layers 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco_r50 --epoch 150 --resume outputs/refcoco_r50/checkpoint0119.pth
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --batch_size 4 --aug_crop --aug_scale --aug_translate --bert_enc_num 12 --vl_enc_layers 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco --epoch 90 --imsize 320


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco_plus_r50 --epochs 180 --lr_drop 120


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50


# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 32 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50
