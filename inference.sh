CLIP_CKPT=$1
ACQ_CKPT=$2
TEST_SET=$3
OUT=$4

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --is_clip --acquirer --new_embed --m_acquirer \
    --clip_ckpt $CLIP_CKPT \
    --acquirer_ckpt $ACQ_CKPT \
    --embedding_ckpt $ACQ_CKPT \
    --img_type $TEST_SET \
    --output_dir $OUT 