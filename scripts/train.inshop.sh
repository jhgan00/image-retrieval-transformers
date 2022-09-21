# set data_path
data_path=/data/In-shop

# IRT_R
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 35000 \
  --dataset inshop \
  --data-path $data_path \
  --m 2 \
  --rank 1 10 20 30 \
  --memory-ratio 0.2 \
  --logging-freq 50 \
  --encoder-momentum 0.999 \
  --lambda-reg 0.7

# IRT_R
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 35000 \
  --dataset inshop \
  --data-path $data_path \
  --m 2 \
  --rank 1 10 20 30 \
  --memory-ratio 0.2 \
  --device cuda:3 \
  --logging-freq 50 \
  --encoder-momentum 0.999 \
  --lambda-reg 0.7

# IRT_L
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 35000 \
  --dataset inshop \
  --data-path $data_path \
  --m 2 \
  --rank 1 10 20 30 \
  --memory-ratio 0.2 \
  --device cuda:3 \
  --logging-freq 50 \
  --encoder-momentum 0.999 \
  --lambda-reg 0.0

# IRT_O
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 0 \
  --dataset inshop \
  --data-path $data_path \
  --rank 1 10 20 30 \
  --logging-freq 50 \
  --memory-ratio 0.2

