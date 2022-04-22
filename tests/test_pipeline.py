from odcl import pipeline
from odcl.inference import BBox
import pytest


@pytest.fixture
def pipeline_f_nodraw_cpu(interpreter_f, tiler_f, color_f):
    """a fixture that returns a pipeline object with no drawing"""
    return pipeline.Pipeline(interpreter_f, tiler_f, color_f, drawer=None)


@pytest.fixture
def pipeline_f_nodraw_tpu(interpreter_f_tpu, tiler_f, color_f):
    """a fixture that returns a pipeline object with no drawing"""
    return pipeline.Pipeline(interpreter_f_tpu, tiler_f, color_f, drawer=None)


@pytest.mark.slow
@pytest.mark.cpu
def test_inference(pipeline_f_nodraw_cpu, raw_image_f):
    """test inference"""
    pipeline_f_nodraw_cpu.inference_over_tiles(raw_image_f)


@pytest.mark.tpu
def test_inference_tpu(pipeline_f_nodraw_tpu, raw_image_f):
    """test inference"""
    pipeline_f_nodraw_tpu.inference_over_tiles(raw_image_f)


@pytest.mark.cpu
def test_inference_small(pipeline_f_nodraw_cpu, raw_image_f_small):
    """test inference"""
    pass
    # pipeline_f_nodraw_cpu.inference_over_tiles(raw_image_f_small)
