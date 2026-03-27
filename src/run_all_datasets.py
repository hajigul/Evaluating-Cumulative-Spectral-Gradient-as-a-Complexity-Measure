from src.main import process_dataset

datasets = ["Nations", "FB15k-237", "WN18RR", "CoDEx-S", ...]  # add your datasets

for ds in datasets:
    try:
        process_dataset(ds)
    except Exception as e:
        print(f"Failed on {ds}: {e}")