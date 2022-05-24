import pytest
from uavfpy.planner.coverage import polygon
import networkx as nx

@pytest.fixture()
def rand_poly():
    return polygon.RandomPolygon(50, holes= 5)

def test_beta_clusters():
    npoints = 50
    points = polygon.beta_clusters(5, ppc = npoints // 5)
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