# set data_path
data_path=/data/Stanford_Online_Products

# IRT_R
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 35000 \
  --dataset sop \
  --m 2 \
  --data-path $data_path \
  --rank 1 10 100 1000 \
  --device cuda:1 \
  --logging-freq 50 \
  --lambda-reg 0.7

# IRT_R
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 35000 \
  --dataset sop \
  --data-path $data_path \
  --m 2 \
  --rank 1 10 100 1000 \
  --device cuda:1 \
  --logging-freq 50 \
  --lambda-reg 0.7

# IRT_L
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 35000 \
  --dataset sop \
  --data-path $data_path \
  --m 2 \
  --rank 1 10 100 1000 \
  --device cuda:1 \
  --logging-freq 50 \
  --lambda-reg 0.0

# IRT_O
python main.py \
  --model deit_small_patch16_224 \
  --max-iter 0 \
  --dataset sop \
  --data-path $data_path \
  --rank 1 10 100 1000 \
  --device cuda:1
