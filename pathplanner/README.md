# A complete-coverage path planner in Python!

So, this module attempts complete-coverage path planning in a 2-d planar world with holes. The world is given by a list of Mx2 arrays of outer boundaries and hole boundaries. Outer boundaries have to be given in clockwise order; hole boundaries must be given in counter-clockwise order.

The algorithm works roughly as follows:

1. Obtain an undirected graph of the outer and inner boundaries
2. Obtain the boustrophedon decomposition of the world
3. Obtain the Reeb Graph of that boustrophedon decomposition
4. For each closed (Affine or Convex) cell, obtain the straight skeleton
5. Link each straight skeleton via midpoints of joined cells

TBD:

6. Traverse the Reeb graph via the straight skeleton.
7. For unvisited cells, perform a boustrophedon sweep of the cell.
8. For visited cells, travel through them via the straight-skeleton path.
9. Continue 7-8 until entire world has been traversed.

## Installation
1. Install python packages in `requirements.txt`.
2. Install `polyskel` via git:

```git clone https://github.com/Botffy/polyskel```

Test with `test.py`