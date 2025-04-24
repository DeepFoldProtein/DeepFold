# Inference

```sh
CUDA_VISIBLE_DEVICES=1 python deepfold/predict.py \
    --input_features_filepath out/7QU2.pkl \
    --output_dirpath out/test_01 \
    --params_dirpath /gpfs/database/casp16/params \
    --preset params_model_1_multimer_v3 \
    --force
```
