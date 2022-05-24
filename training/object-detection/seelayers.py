import tensorflow as tf
import re

CKPT_PATH = "./checkpoints/ckpts01/ckpt-214"
ckpt_reader = tf.train.load_checkpoint(CKPT_PATH)
vars_list = tf.train.list_variables(CKPT_PATH)
for v in vars_list:
    printed_name = ""
    for part in v[0].split("/")[:2]:
        printed_name += part + " / "
    regex = re.compile(r"(efficientnet-lite1.Sblocks_)")
    if re.search(regex, printed_name) is not None:
        print(printed_name)
