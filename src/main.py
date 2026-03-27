import argparse
import sys
import os

from .config import DEVICE
from .data_loader import load_and_group_by_tail
from .embeddings import compute_hr_embeddings
from .similarity import build_class_mean_vectors
from .spectral import compute_class_overlap, compute_csg
from .utils import get_dataset_file_paths, save_csg_result


def process_dataset(dataset_name, base_dir="data"):
    print(f"\n=== Processing dataset: {dataset_name} ===")

    file_paths = get_dataset_file_paths(dataset_name, base_dir)
    triplets, tail_to_hr = load_and_group_by_tail(file_paths)

    unique_heads = sorted({h for h, _, _ in triplets})
    unique_relations = sorted({r for _, r, _ in triplets})

    head_emb, rel_emb = compute_hr_embeddings(triplets, unique_heads, unique_relations)

    class_vectors, _ = build_class_mean_vectors(
        triplets, head_emb, rel_emb, tail_to_hr
    )

    print(f"Number of classes (unique tails): {len(class_vectors)}")

    similarity_matrix = compute_class_overlap(class_vectors)
    csg = compute_csg(similarity_matrix)

    print(f"CSG for {dataset_name}: {csg:.6f}")
    save_csg_result(dataset_name, csg)

    return csg


def main():
    parser = argparse.ArgumentParser(description="Compute CSG complexity for KG datasets")
    parser.add_argument("--dataset", type=str, default="Nations",
                        help="Dataset name (folder name under data/)")
    parser.add_argument("--base_dir", type=str, default="data",
                        help="Base directory containing dataset folders")

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.base_dir, args.dataset)):
        print(f"Error: Dataset folder '{args.dataset}' not found in {args.base_dir}")
        sys.exit(1)

    process_dataset(args.dataset, args.base_dir)


if __name__ == "__main__":
    main()