import pytest
from uavfpy.planner.coverage import polygon
import networkx as nx
import numpy as np

# a boundary whose points are oriented clockwise
CW = [
    [0.1, 0.2],
    [0.6, 0.6],
    [0.8, 0.1],
    [0.3, -0.5],
    [0.15, -0.1],
]

# a hole whose points are oriented clockwise
CW_HOLE = [
    [0.2, 0.2],
    [0.4, 0.25],
    [0.3, 0.0],
]


@pytest.fixture()
def rand_poly():
    return polygon.RandomPolygon(50, holes=5)


def test_beta_clusters():
    npoints = 50
    points = polygon.beta_clusters(5, ppc=npoints // 5)
    assert points.shape[0] == npoints
    assert points.shape[1] == 2


def test_make_rand_poly(rand_poly):
    assert isinstance(rand_poly.G, nx.DiGraph)


def test_randpoly_properties(rand_poly):
    # planarity
    assert nx.is_planar(rand_poly.G)

    # connectivity
    weakly_connected_parts = list(nx.weakly_connected_components(rand_poly.G))
    if rand_poly.nholes == 0:
        assert nx.is_weakly_connected(rand_poly.G)
    else:
        assert len(weakly_connected_parts) == rand_poly.nholes + 1

    # the graph contains one cycle for the outer boundary, and one cycle for each hole
    assert len(list(nx.simple_cycles(rand_poly.G))) == rand_poly.nholes + 1


@pytest.fixture()
def boundaries_cw():
    boundaries = np.array(CW)
    return boundaries


@pytest.fixture()
def boundaries_ccw():
    boundaries = np.array(list(reversed(CW)))
    return boundaries


@pytest.fixture()
def holes_cw():
    holes = [np.array(CW_HOLE)]
    return holes


@pytest.fixture()
def holes_ccw():
    holes = [np.array(list(reversed(CW_HOLE)))]
    return holes
