import os

def get_dataset_file_paths(dataset_name, base_dir="data"):
    data_dir = os.path.join(base_dir, dataset_name)
    return [
        os.path.join(data_dir, "train.txt"),
        os.path.join(data_dir, "valid.txt"),
        os.path.join(data_dir, "test.txt"),
    ]


def save_csg_result(dataset_name, csg_value, base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    output_path = os.path.join(base_dir, f"{dataset_name}_csg.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"CSG Measure: {csg_value:.6f}\n")
    print(f"CSG result saved to: {output_path}")