import os, shutil
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path


def prune(indir, odir, prune_every, dry_run=False):
    indir = Path(indir)
    odir = Path(odir)

    os.makedirs(odir, exist_ok=True)
    os.makedirs(odir / "images", exist_ok=True)
    os.makedirs(odir / "labels", exist_ok=True)
    with open(indir / "traininglist.txt", "r") as train, open(
        indir / "testlist.txt", "r"
    ) as test:
        for i, label in zip([train, test], ["train", "test"]):
            n = 0
            f = open(str((odir / label).resolve()) + ".txt", "w")
            for a, j in tqdm(enumerate(i.readlines()), desc=label):
                if a % prune_every == 0:
                    n += 1
                    j = str(j)
                    j = j.strip()
                    fname = j.split("/")[-1].split(".")[0]
                    iimg = indir / "images" / (fname + ".jpg")
                    itxt = indir / "labels" / (fname + ".txt")
                    destimg = odir / "images" / (fname + ".jpg")
                    desttxt = odir / "labels" / (fname + ".txt")
                    f.write(str(destimg.resolve()) + "\n")
                    if not dry_run:
                        shutil.copy(iimg, destimg)
                        shutil.copy(itxt, desttxt)
            f.close()


if __name__ == "__main__":
    argp = ArgumentParser(
        description="prune a dataset to reduce its size, by copying only every nth image in the dataset."
    )
    argp.add_argument("-i", type=str, required=True, help="input directory")
    argp.add_argument("-o", type=str, required=True, help="output directory")
    argp.add_argument("-n", type=int, required=True, help="keep every n images")
    argp.add_argument(
        "-d", type=bool, required=False, help="dry run. if true, don't copy anything."
    )
    opts = argp.parse_args()
    prune(opts.i, opts.o, opts.n, opts.d)
