# run cd ./projects_xwangfy/nexperia_wxs  bash ./scripts/train/mixup+apex.sh
CUDA_VISIBLE_DEVICES='6'
RECORDED_FILE='09-27_mixup+apex'
TRAIN='mixup'
#IMBALANCED_SAMPLER=1 # 可以通过默认的是False但是在这里进行启动使得变为True 但是需要注意不管这里是等于0还是1都是True了
BATCH_SIZE=128
MODEL='resnet50'
python -u main_plus_apex.py --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} --batch_size ${BATCH_SIZE} \
--recording_file ${RECORDED_FILE} --train ${TRAIN} --model ${MODEL}