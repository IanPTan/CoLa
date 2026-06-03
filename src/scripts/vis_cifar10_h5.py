import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Visualize CIFAR-10 images from HDF5 file.")
    parser.add_argument("h5_path", help="Path to the CIFAR-10 HDF5 file")
    parser.add_argument("indices", type=int, nargs="*", help="Indices of images to visualize")
    parser.add_argument("--test", action="store_true", help="Use test split instead of train")

    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"Error: File {args.h5_path} not found.")
        return

    split = "test" if args.test else "train"

    with h5py.File(args.h5_path, "r") as f:
        # Load label names if they exist
        label_names = []
        if "label_names" in f:
            label_names = [name.decode('utf-8') for name in f["label_names"][:]]

        # Print dataset statistics
        print("HDF5 Dataset Info:")
        for s in ["train", "test"]:
            if s in f:
                img_shape = f[s]["images"].shape
                lbl_shape = f[s]["labels"].shape
                print(f"  - {s.capitalize()} images: {img_shape}")
                print(f"  - {s.capitalize()} labels: {lbl_shape}")
            else:
                print(f"  - {s.capitalize()} split not found.")
        
        if label_names:
            print(f"  - Label names: {', '.join(label_names)}")

        if not args.indices:
            return

        print(f"\nVisualizing {len(args.indices)} images from '{split}' split...")

        images = []
        labels = []
        for idx in args.indices:
            if idx < 0 or idx >= f[split]["images"].shape[0]:
                print(f"Warning: Index {idx} is out of bounds for {split} split.")
                continue
            
            # (3, 32, 32) uint8 -> (32, 32, 3) for matplotlib
            img = f[split]["images"][idx]
            img = np.transpose(img, (1, 2, 0))
            lbl_idx = f[split]["labels"][idx]
            lbl_name = label_names[lbl_idx] if label_names else "Unknown"
            
            images.append(img)
            labels.append((idx, lbl_idx, lbl_name))

        if not images:
            return

        # Display images
        num_images = len(images)
        if num_images == 1:
            idx, lbl_idx, lbl_name = labels[0]
            plt.figure(figsize=(4, 4))
            plt.imshow(images[0])
            plt.title(f"Split: {split} | Idx: {idx}\nLabel: {lbl_idx} ({lbl_name})")
            plt.axis("off")
        else:
            cols = min(num_images, 5)
            rows = (num_images + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            axes = axes.flatten() if num_images > 1 else [axes]
            
            for i, (img, (idx, lbl_idx, lbl_name)) in enumerate(zip(images, labels)):
                axes[i].imshow(img)
                axes[i].set_title(f"Idx: {idx}\n{lbl_idx} ({lbl_name})")
                axes[i].axis("off")
            
            # Hide empty subplots
            for i in range(num_images, len(axes)):
                axes[i].axis("off")
            
            plt.tight_layout()
        
        plt.show()


if __name__ == "__main__":
    main()
