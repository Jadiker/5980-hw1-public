# NLP HW1 Instructions
This model was trained using UMN NLP's Grace computer.
I used Docker with the nvcr.io/nvidia/pytorch:21.10-py3 image.
Then, run `pip install datasets transformers wandb && wandb login`.
This will prompt you to login to Weights and Biases for logging.

Then, to do the training, run
`python3 run_glue.py --task_name sst2 --do_train --do_eval --model_name roberta-large --overwrite_output_dir --evaluation_strategy steps --adam_beta1=0.9 --adam_beta2=0.98 --adam_epsilon=1e-6 --weight_decay=0.01 --warmup_steps=30000 --learning_rate=2e-05 --run_name roberta_sst2 --output_dir roberta_sst2`

Then, use `download_dataset` to get the validation data that we're going to test with.
(Note that because we're doing SST2, we can't actually evaluate on the real test data. See [this](https://github.com/huggingface/datasets/issues/245) for more details.)

Then, move the checkpoint with the highest accuracy into an outer folder called `final_models`.

Then, run `python run_glue.py --train_file val_dataset.csv --validation_file val_dataset.csv --test_file val_dataset.csv --do_predict --model_name final_models/checkpoint-14000 --run_name roberta-sst2-val --output_dir roberta-sst2-val` (with `checkpoint-14000` replaced with your best checkpoint)

(This may look odd, but the script requires a train and validation file, even if it doesn't use them.)

You should find the predictions (and statistics) in the newly-created folder called `roberta-sst2-val`.