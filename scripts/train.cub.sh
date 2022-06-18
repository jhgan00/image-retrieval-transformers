# IRT_O
#python main.py \
#  --model deit_small_patch16_224 \
#  --max-iter 0 \
#  --data-set cub200 \
#  --data-path ./data/CUB_200_2011 \
#  --rank 1 2 4 8

# IRT_L
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 2000 \
  --data-set cub200 \
  --data-path ./data/CUB_200_2011 \
  --rank 1 2 4 8 \
  --lambda-reg 0

# IRT_R
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 2000 \
  --data-set cub200 \
  --data-path ./data/CUB_200_2011 \
  --rank 1 2 4 8 \
  --lambda-reg 0.7 \
  --device cuda:1

# IRT_R
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 2000 \
  --data-set cub200 \
  --data-path ./data/CUB_200_2011 \
  --rank 1 2 4 8 \
  --lambda-reg 0.7 \
  --device cuda:2
