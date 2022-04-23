import os, shutil
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path


def movef(idir, odir, verbose=False, dry_run=False):
    idir = Path(idir).resolve()
    odir = Path(odir).resolve()
    imglabs = ["images", "labels"]
    folders = ["test", "train"]
    os.makedirs(odir / "images", exist_ok=True)
    os.makedirs(odir / "labels", exist_ok=True)

    with open(str(odir) + "/traininglist.txt", "w") as trainl, open(
        str(odir) + "/testlist.txt", "w"
    ) as testl:
        for i in imglabs:
            for j in folders:
                for k in tqdm(os.listdir(idir / i / j), desc=i + "," + j):
                    _, ext = k.split(".")
                    # read out all images
                    src = idir / i / j / k
                    # only show path to jpg images
                    if ext == "jpg":
                        dst = odir / "images" / k
                        # write train, test, val
                        if j == "train":
                            trainl.write(str(odir / "images" / k) + "\n")
                        elif j == "test":
                            testl.write(str(odir / "images" / k) + "\n")
                    # copy txt files to labels dir
                    elif ext == "txt":
                        dst = (odir / "labels" / k).resolve()
                    # perform move
                    verbose_output = str(src) + "-->" + str(dst)
                    if dry_run:
                        pass
                        if verbose:
                            print(verbose_output)
                    else:
                        shutil.copy(str(src), str(dst))
                        if verbose:
                            print(verbose_output)


if __name__ == "__main__":
    argp = ArgumentParser(
        description="take a (split!) YOLO formatted dataset and prepare for reformatting to TFRECORD."
    )
    argp.add_argument("-i", type=str, required=True, help="input directory")
    argp.add_argument("-o", type=str, required=True, help="output directory")
    argp.add_argument(
        "-v",
        type=str,
        required=False,
        help="verbose. print each copied file to console",
    )
    argp.add_argument(
        "-d", type=str, required=False, help="dry run. If true, doesn't copy anything."
    )
    opts = argp.parse_args()
    movef(opts.i, opts.o)
