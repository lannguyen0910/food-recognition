import argparse
import random
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('-folder', type=str, default='.',
                    help='path to images folder')
parser.add_argument('-ratio', type=float, default=0.9,
                    help='ratio of the train set (default: 0.9)')
parser.add_argument('-seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('-out', type=str, default='.',
                    help='directory to save the splits (default: .)')

if __name__ == '__main__':
    args = parser.parse_args()

    # Seed the random processes
    random.seed(args.seed)

    classes = os.listdir(args.folder)

    for i in ["train", "val"]:
        path = "_".join([args.folder, i])
        if not os.path.exists(path):
            os.mkdir(path)
            for j in classes:
                new_path = os.path.join(path, j)
                os.mkdir(new_path)

    total_train = 0
    total_val = 0

    print("Start spliting folder " + args.folder + "...")
    for i in classes:
        path = os.path.join(args.folder, i)
        img_path = list(os.listdir(path))
        random.shuffle(img_path)

        num_image = len(img_path)
        num_train = int(num_image*args.ratio)

        train_imgs = img_path[:num_train]
        val_imgs = img_path[num_train:]

        for img in tqdm(train_imgs):
            total_train += 1
            src_path = os.path.join(args.folder, i, img)
            tgt_path = os.path.join("_".join([args.folder, "train"]), i, img)
            try:
                os.rename(src_path, tgt_path)
            except Exception as E:
                print(src_path + " not found")

        for img in tqdm(val_imgs):
            total_val += 1
            src_path = os.path.join(args.folder, i, img)
            tgt_path = os.path.join("_".join([args.folder, "val"]), i, img)
            try:
                os.rename(src_path, tgt_path)
            except Exception as E:
                print(src_path + " not found")
    print("_______________________________________________")
    print("Number of training images: " + str(total_train))
    print("Number of validation images: " + str(total_val))
