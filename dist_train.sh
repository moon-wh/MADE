# train
# prcc
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train.py --config_file configs/prcc/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True
# ltcc
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train.py --config_file configs/ltcc/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True
# Celeb_reID_light
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train.py --config_file configs/Celeb_light/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True
# last
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train.py --config_file configs/last/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True
# test
#CUDA_VISIBLE_DEVICES=1,0 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 test.py --config_file configs/ltcc/eva02_l_maskmeta_random.yml


