import pytest, cv2
from pathlib import Path
from uavfpy.odcl import inference, color

MODEL_PATH = "../example/efficientdet_lite0_320_ptq.tflite"
MODEL_PATH_TPU = "../example/efficientdet_lite0_320_ptq_edgetpu.tflite"
LABEL_PATH = "../example/coco_labels.txt"
IMG_PATH = "../example/plaza.jpg"
MODEL_PATH = str((Path(__file__).parent / MODEL_PATH).resolve())
MODEL_PATH_TPU = str((Path(__file__).parent / MODEL_PATH_TPU).resolve())
LABEL_PATH = str((Path(__file__).parent / LABEL_PATH).resolve())
IMG_PATH = str((Path(__file__).parent / IMG_PATH).resolve())

"""By default, pytest will run all tests. But, with a CPU-only machine, running even one high-resolution image
through the pipeline can take quite a long time. So, we will skip these high-resolution tests, and run pytest at
a lower resolution by default, unless this flag is set by the user."""


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Pass to run slow tests. WARNING: it will take a long time (several minutes) to run tests with this option passed!",
    )
    parser.addoption(
        "--tpu",
        action="store_true",
        default=False,
        help="Pass to run tests on TPU. Will run slow tests on TPU.",
    )
    parser.addoption(
        "--skipdnn",
        action="store_true",
        default=False,
        help="Pass to skip dnn tests. No tflite/pipeline/edgetpu tests will run with this flag passed",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "tpu: mark test as running on TPU")
    config.addinivalue_line("markers", "cpu: mark test as running on CPU")
    config.addinivalue_line("markers", "dnn: mark test as running on DNN")


def pytest_collection_modifyitems(config, items):
    slow = config.getoption("--slow")
    tpu = config.getoption("--tpu")
    skipdnn = config.getoption("--skipdnn")
    if skipdnn:
        skip_dnn = pytest.mark.skip(reason="--skipdnn flag passed")
        for item in items:
            if "dnn" in item.keywords:
                item.add_marker(skip_dnn)
    else:
        if slow and tpu:
            # run both cpu and tpu tests on high res
            return
        elif slow and not tpu:
            # run slow tests on cpu
            skip_tpu = pytest.mark.skip(reason="--tpu flag not passed, skipping tpu tests")
            for item in items:
                if "tpu" in item.keywords:
                    item.add_marker(skip_tpu)
        elif tpu and not slow:
            # run slow tests on tpu
            skip_cpu = pytest.mark.skip(reason="--tpu flag passed, skipping cpu tests")
            for item in items:
                if "cpu" in item.keywords:
                    item.add_marker(skip_cpu)
        else:
            skip_slow = pytest.mark.skip(reason="--slow flag passed, skipping slow tests")
            skip_tpu = pytest.mark.skip(reason="--tpu flag not passed, skipping tpu tests")
            for item in items:
                if "slow" in item.keywords:
                    item.add_marker(skip_slow)
                if "tpu" in item.keywords:
                    item.add_marker(skip_tpu)


##########################################################
######              fixtures                   ###########
##########################################################


@pytest.fixture
def interpreter_f():
    return inference.TargetInterpreter(
        MODEL_PATH, LABEL_PATH, "cpu", 0.4, order_key="efficientdetd0"
    )


@pytest.fixture
def interpreter_f_tpu():
    return inference.TargetInterpreter(
        MODEL_PATH_TPU, LABEL_PATH, "tpu", 0.4, order_key="efficientdetd0"
    )


@pytest.fixture
def tiler_f():
    return inference.Tiler(320, 50)


@pytest.fixture
def color_f():
    return color.Color()


@pytest.fixture
def raw_image_f():
    return cv2.imread(IMG_PATH)


@pytest.fixture
def raw_image_f_small():
    image = cv2.imread(IMG_PATH)
    print(image.shape)
