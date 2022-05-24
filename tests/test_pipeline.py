from uavfpy.odcl import pipeline
from uavfpy.odcl.inference import BBox
import pytest

"""EVERYTHING THAT RUNS TFLITE INFERENCE GETS MARKED DNN"""


@pytest.fixture
@pytest.mark.dnn
def pipeline_f_nodraw_cpu(interpreter_f, tiler_f, color_f):
    """a fixture that returns a pipeline object with no drawing"""
    return pipeline.Pipeline(interpreter_f, tiler_f, color_f, drawer=None)


@pytest.fixture
@pytest.mark.dnn
def pipeline_f_nodraw_tpu(interpreter_f_tpu, tiler_f, color_f):
    """a fixture that returns a pipeline object with no drawing"""
    return pipeline.Pipeline(interpreter_f_tpu, tiler_f, color_f, drawer=None)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.dnn
def test_inference(pipeline_f_nodraw_cpu, raw_image_f):
    """test inference"""
    pipeline_f_nodraw_cpu.inference_over_tiles(raw_image_f)


@pytest.mark.tpu
@pytest.mark.dnn
def test_inference_tpu(pipeline_f_nodraw_tpu, raw_image_f):
    """test inference"""
    pipeline_f_nodraw_tpu.inference_over_tiles(raw_image_f)


@pytest.mark.cpu
@pytest.mark.dnn
def test_inference_small(pipeline_f_nodraw_cpu, raw_image_f_small):
    """test inference"""
    pass
    # pipeline_f_nodraw_cpu.inference_over_tiles(raw_image_f_small)
