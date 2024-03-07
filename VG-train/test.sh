export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5


# ReferItGame
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./released_models/TransVG_referit.pth --output_dir ./outputs/referit_r50


# # # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --use_mae --batch_size 64 --num_workers 4 --dataset unc --max_query_len 20 --eval_set testA --imsize 320 --eval_model outputs/refcoco_20x20_vl256/best_checkpoint.pth --output_dir ./outputs/refcoco_20x20_vl256
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --use_mae --batch_size 64 --num_workers 4 --dataset unc --max_query_len 20 --eval_set testB --imsize 320 --eval_model outputs/refcoco_20x20_vl256/best_checkpoint.pth --output_dir ./outputs/refcoco_20x20_vl256


# # # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --use_mae --batch_size 64 --num_workers 4 --dataset unc+ --max_query_len 20 --eval_set testA --imsize 320 --eval_model outputs/refcoco+_20x20_vl256/best_checkpoint.pth --output_dir ./outputs/refcoco+_20x20_vl256
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --use_mae --batch_size 64 --num_workers 4 --dataset unc+ --max_query_len 20 --eval_set testB --imsize 320 --eval_model outputs/refcoco+_20x20_vl256/best_checkpoint.pth --output_dir ./outputs/refcoco+_20x20_vl256


# # # RefCOCOg u-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --use_mae --batch_size 64 --num_workers 4 --dataset gref_umd --max_query_len 40 --eval_set test --imsize 320 --eval_model outputs/refcocog_20x20_vl256/best_checkpoint.pth --output_dir ./outputs/refcocog_20x20_vl256

python eval.py --use_mae --batch_size 1 --num_workers 1 --dataset unc --max_query_len 21 --eval_set testA --imsize 640 --eval_model outputs/refcoco_sfa/best_checkpoint.pth --output_dir outputs/refcoco_sfa/