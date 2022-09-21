# set data_path
data_path=/data/CUB_200_2011

# IRT_O
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 0 \
  --dataset cub200 \
  --data-path $data_path \
  --rank 1 2 4 8 \
  --logging-freq 10 \
  --device cuda:0

# IRT_L
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 2000 \
  --dataset cub200 \
  --data-path $data_path \
  --rank 1 2 4 8 \
  --lambda-reg 0 \
  --logging-freq 10 \
  --device cuda:0

## IRT_R
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 2000 \
  --dataset cub200 \
  --data-path $data_path \
  --rank 1 2 4 8 \
  --lambda-reg 0.7 \
  --logging-freq 10 \
  --device cuda:0

# IRT_R
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 2000 \
  --dataset cub200 \
  --data-path $data_path \
  --rank 1 2 4 8 \
  --lambda-reg 0.7 \
  --logging-freq 10 \
  --device cuda:0
