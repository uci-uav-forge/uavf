from sklearn.model_selection import train_test_split
import os, shutil
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def tts(input_dir, output_dir, test_size, verbose=False, dry_run=False):
    idir = Path(input_dir)
    odir = Path(output_dir)

    imgflist = []
    for img in os.listdir(idir / "images"):
        imgpath = idir / "images" / img
        txtpath = idir / "labels" / (img[:-4] + ".txt")
        imgflist.append((imgpath, txtpath))

    train, test = train_test_split(imgflist, test_size=test_size)

    for tt, ttf in zip(["train", "test"], (train, test)):
        # training directories
        os.makedirs(odir / "images" / tt, exist_ok=True)
        os.makedirs(odir / "labels" / tt, exist_ok=True)

        for imagepath, labelpath in tqdm(ttf, desc="copy " + tt + " set"):
            copyto_imagepath = (
                str((odir / "images" / tt).resolve())
                + "/"
                + str(imagepath).split("/")[-1]
            )
            copyto_labelpath = (
                str((odir / "labels" / tt).resolve())
                + "/"
                + str(labelpath).split("/")[-1]
            )
            if dry_run:
                pass
                if verbose:
                    print(imagepath, "-->", copyto_imagepath)
                    print(labelpath, "-->", copyto_labelpath)
            else:
                shutil.copy(imagepath, copyto_imagepath)
                shutil.copy(labelpath, copyto_labelpath)
                if verbose:
                    print(imagepath, "-->", copyto_imagepath)
                    print(labelpath, "-->", copyto_labelpath)


if __name__ == "__main__":
    argp = ArgumentParser(
        description="take a YOLO formatted dataset and split it into train, val sets."
    )
    argp.add_argument("-i", type=str, required=True, help="input directory")
    argp.add_argument("-o", type=str, required=True, help="output directory")
    argp.add_argument(
        "-t",
        type=float,
        required=True,
        help="test size, as percentage of original dataset. Give as a number between 0, 1.",
    )
    argp.add_argument(
        "-v",
        type=str,
        required=False,
        help="verbose. if true, print each copied file to console",
    )
    argp.add_argument(
        "-d", type=str, required=False, help="dry run. If true, doesn't copy anything."
    )
    opts = argp.parse_args()
    tts(opts.i, opts.o, opts.t, opts.v, opts.d)
