# IRT_O
#python main.py \
#  --model deit_small_patch16_224 \
#  --max-iter 0 \
#  --data-set sop \
#  --data-path ./data/Stanford_Online_Products \
#  --rank 1 10 100 1000 \
#  --memory-ratio 0.2 \
#  --device cuda:1

# IRT_R
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 35000 \
  --data-set sop \
  --data-path ./data/Stanford_Online_Products \
  --rank 1 10 100 1000 \
  --device cuda:1 \
  --lambda-reg 0.7

# IRT_R
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 35000 \
  --data-set sop \
  --data-path ./data/Stanford_Online_Products \
  --rank 1 10 100 1000 \
  --device cuda:1 \
  --lambda-reg 0.7

# IRT_L
#python main.py \
#  --model deit_small_patch16_224 \
#  --max-iter 35000 \
#  --data-set sop \
#  --data-path ./data/Stanford_Online_Products \
#  --rank 1 10 100 1000 \
#  --device cuda:1 \
#  --lambda-reg 0.0

