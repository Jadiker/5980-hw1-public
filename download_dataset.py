
from datasets import load_dataset
raw_datasets = load_dataset(
    "glue",
    "sst2",
    cache_dir="./downloaded_sst2"
)

# get the validation dataset
val_dataset = raw_datasets["validation"]

# save the val dataset to a csv file
val_dataset.to_csv("val_dataset.csv", index=False)