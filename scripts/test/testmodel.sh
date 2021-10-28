# run: cd ./projects_xwangfy/nexperia_wxs    bash ./scripts/test/testmodel.sh
RECORDED_FILE='09-27_base+apex'
CUDA_VISIBLE_DEVICES='4'
MODEL='resnet50'
for TEST_DATA in Jan Feb Mar
do
  for MODEL_SAVED in _best1.pth _best2.pth _best3.pth _epoch148.pth _epoch149.pth _epoch150.pth
  do
    python -u tester.py --recorded_file ${RECORDED_FILE} \
    --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
    --model ${MODEL} \
    --test_data ${TEST_DATA} \
    --model_saved ${MODEL_SAVED}
  done
done