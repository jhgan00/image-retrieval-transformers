# IRT_O
#python main.py \
#  --model deit_small_patch16_224 \
#  --max-iter 0 \
#  --data-set inshop \
#  --data-path ./data/In-shop \
#  --rank 1 10 20 30 \
#  --memory-ratio 0.2 \
#  --device cuda:2

# IRT_R
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 35000 \
  --data-set inshop \
  --data-path ./data/In-shop \
  --rank 1 10 20 30 \
  --memory-ratio 0.2 \
  --device cuda:2 \
  --encoder-momentum 0.999 \
  --lambda-reg 0.7

# IRT_R
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 35000 \
  --data-set inshop \
  --data-path ./data/In-shop \
  --rank 1 10 20 30 \
  --memory-ratio 0.2 \
  --device cuda:2 \
  --encoder-momentum 0.999 \
  --lambda-reg 0.7

# IRT_L
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 35000 \
  --data-set inshop \
  --data-path ./data/In-shop \
  --rank 1 10 20 30 \
  --memory-ratio 0.2 \
  --device cuda:2 \
  --encoder-momentum 0.999 \
  --lambda-reg 0.0

