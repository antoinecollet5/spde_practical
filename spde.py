"""Implement functions to perform geostatistics from the spde approach."""
from typing import Callable, Optional, Union

import numpy as np
import scipy
from pyrtid.utils.grid import span_to_node_numbers_3d
from pyrtid.utils.types import NDArrayFloat, NDArrayInt
from scipy._lib._util import check_random_state  # To handle random_state
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import LinearOperator, splu
from sklearn.gaussian_process.kernels import Matern as skMatern
from sksparse.cholmod import Factor, cholesky

Int = Union[int, NDArrayInt]


def indices_to_node_number(
    ix: Int,
    nx: int = 1,
    iy: Int = 0,
    ny: int = 1,
    iz: Int = 0,
    indices_start_at_one: bool = False,
) -> Int:
    """
    Convert indices (ix, iy, iz) to a node-number.

    For 1D and 2D, simply leave iy, ny, iz and nz to their default values.

    Note
    ----
    Node numbering start at zero.

    Warning
    -------
    This applies only for regular grids. It is not suited for vertex.

    Parameters
    ----------
    ix : int
        Index on the x-axis.
    nx : int, optional
        Number of meshes on the x-axis. The default is 1.
    iy : int, optional
        Index on the y-axis. The default is 0.
    ny : int, optional
        Number of meshes on the y-axis. The default is 1.
    iz : int, optional
        Index on the z-axis. The default is 0.
    indices_start_at_one: bool, optional
        Whether the indices start at 1. Otherwise, start at 0. The default is False.

    Returns
    -------
    int
        The node number.

    """
    if indices_start_at_one:
        ix = np.max((ix - 1, 0))
        iy = np.max((iy - 1, 0))
        iz = np.max((iz - 1, 0))
    return ix + (iy * nx) + (iz * ny * nx)


def get_laplacian_matrix_for_loops(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    kappa: Union[NDArrayFloat, float],
) -> csc_matrix:
    """
    Return a sparse matrix of the discretization of the Laplacian.

    Note
    ----
    This is a very inefficient implementation which is simply dedicated to check the
    vectorial implementation correctness. This could be interesting when sparse objects
    are supported by numba with the jit compiler.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz : int
        Number of meshes along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : float
        Range (length scale).

    Returns
    -------
    csc_matrix
        Sparse matrix with dimension (nx * ny)x(nx * ny) representing the  discretized
        laplacian.

    """
    n_nodes = nx * ny * nz
    if np.isscalar(kappa):
        _kappa = np.full(n_nodes, fill_value=kappa)
    else:
        _kappa = np.array(kappa).ravel("F")
    # construct an empty sparse matrix (lil_format because it supports indexing and
    # slicing).
    lap = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

    # Looping on all nodes and considering neighbours
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                node_index = int(indices_to_node_number(ix, nx, iy, ny, iz))
                lap[node_index, node_index] += _kappa[node_index] ** 2

                if nx > 1:
                    lap[node_index, node_index] += 2 / dx**2
                if ny > 1:
                    lap[node_index, node_index] += 2 / dy**2
                if nz > 1:
                    lap[node_index, node_index] += 2 / dz**2

                # X contribution
                if ix > 0:
                    neighbor_index = int(indices_to_node_number(ix - 1, nx, iy, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dx**2
                if ix < nx - 1:
                    neighbor_index = int(indices_to_node_number(ix + 1, nx, iy, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dx**2

                # Y contribution
                if iy > 0:
                    neighbor_index = int(indices_to_node_number(ix, nx, iy - 1, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dy**2
                if iy < ny - 1:
                    neighbor_index = int(indices_to_node_number(ix, nx, iy + 1, ny, iz))
                    lap[node_index, neighbor_index] += -1.0 / dy**2

                # Z contribution
                if iz > 0:
                    neighbor_index = int(indices_to_node_number(ix, nx, iy, ny, iz - 1))
                    lap[node_index, neighbor_index] += -1.0 / dz**2
                if iz < nz - 1:
                    neighbor_index = int(
                        indices_to_node_number(
                            ix,
                            nx,
                            iy,
                            ny,
                            iz + 1,
                        )
                    )
                    lap[node_index, neighbor_index] += -1.0 / dz**2

    # Convert from lil to csr matrix for more efficient calculation
    return lap.tocsc()


def get_laplacian_matrix(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    kappa: Union[NDArrayFloat, float],
) -> csc_matrix:
    """
    Return a sparse matrix of the discretization of the Laplacian.

    Note
    ----
    This should be a bit more efficient than the for-loop version for large datasets.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz : int
        Number of meshes along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : float
        Range (length scale).

    Returns
    -------
    csc_matrix
        Sparse matrix with dimension (nx * ny)x(nx * ny) representing the  discretized
        laplacian.

    """
    n_nodes = nx * ny * nz
    if np.isscalar(kappa):
        _kappa = np.full(n_nodes, fill_value=kappa)
    else:
        _kappa = np.array(kappa).ravel("F")
    # construct an empty sparse matrix (lil_format because it supports indexing and
    # slicing).
    lap = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

    # Add kappa on the diagonal
    lap.setdiag(lap.diagonal() + _kappa**2)

    # X contribution
    if nx > 1:
        lap.setdiag(lap.diagonal() + 2 / dx**2)
        indices_owner: NDArrayInt = span_to_node_numbers_3d(
            (slice(0, nx - 1), slice(None), slice(None)), nx=nx, ny=ny, nz=nz
        )
        indices_neigh: NDArrayInt = span_to_node_numbers_3d(
            (slice(1, nx), slice(None), slice(None)), nx=nx, ny=ny, nz=nz
        )

        # forward
        lap[indices_owner, indices_neigh] -= np.ones(indices_owner.size) / dx**2
        # backward
        lap[indices_neigh, indices_owner] -= np.ones(indices_owner.size) / dx**2

    # Y contribution
    if ny > 1:
        lap.setdiag(lap.diagonal() + 2 / dy**2)
        indices_owner: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(0, ny - 1), slice(None)), nx=nx, ny=ny, nz=nz
        )
        indices_neigh: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(1, ny), slice(None)), nx=nx, ny=ny, nz=nz
        )

        # forward
        lap[indices_owner, indices_neigh] -= np.ones(indices_owner.size) / dy**2
        # backward
        lap[indices_neigh, indices_owner] -= np.ones(indices_owner.size) / dy**2

    # Z contribution
    if nz > 1:
        lap.setdiag(lap.diagonal() + 2 / dz**2)

        indices_owner: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(None), slice(0, nz - 1)), nx=nx, ny=ny, nz=nz
        )
        indices_neigh: NDArrayInt = span_to_node_numbers_3d(
            (slice(None), slice(None), slice(1, nz)), nx=nx, ny=ny, nz=nz
        )

        # forward
        lap[indices_owner, indices_neigh] -= np.ones(indices_owner.size) / dz**2
        # backward
        lap[indices_neigh, indices_owner] -= np.ones(indices_owner.size) / dz**2

    # Convert from lil to csr matrix for more efficient calculation
    return lap.tocsc()


def get_laplacian_matrix_1d(
    nx: int, dx: float, kappa: Union[NDArrayFloat, float]
) -> csc_matrix:
    """
    Return a sparse matrix of the discretization of the Laplacian.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    dx : float
        Size of the mesh along x.
    kappa : float
        Range (length scale).

    Returns
    -------
    csc_matrix
        Sparse matrix with dimension (nx * ny)x(nx * ny) representing the  discretized
        laplacian.

    """
    n_nodes = nx
    if np.isscalar(kappa):
        _kappa = np.full(n_nodes, fill_value=kappa)
    else:
        _kappa = np.array(kappa)
    # construct an empty sparse matrix (lil_format because it supports indexing and
    # slicing).
    lap = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

    # Looping on all nodes and considering neighbors
    for i in range(nx):
        node_index = i
        lap[node_index, node_index] += _kappa[node_index] ** 2 + 2 / dx**2
        if i > 0:
            neighbor_index = indices_to_node_number(i - 1)
            lap[node_index, neighbor_index] += -1.0 / dx**2
        if i < nx - 1:
            neighbor_index = indices_to_node_number(i + 1)
            lap[node_index, neighbor_index] += -1.0 / dx**2

    # Convert from lil to csr matrix for more efficient calculation
    return lap.tocsc()


def get_laplacian_matrix_2d(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    kappa: Union[NDArrayFloat, float],
    anisotropy_function: Optional[Callable[[float, float], NDArrayFloat]] = None,
) -> csc_matrix:
    """
    Return a sparse matrix of the discretization of the Laplacian.


    See [1] for the details.

    [1] G.-A. Fuglstad, F. Lindgren, D. Simpson, and H. Rue, “Exploring a New Class of
    Non-Stationary Spatial Gaussian Random Fields with Varying Local Anisotropy,”
    Statistica Sinica, vol. 25, no. 1, pp. 115–133, 2015.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    kappa : float
        Range (length scale).
    anisotropy_function: Optional[Callable[[float, float], NDArrayFloat]]
        Anisotropy function that returns a 2d transformation value for a given
        position (x, y). The default is None.

    Returns
    -------
    csc_matrix
        Sparse matrix with dimension (nx * ny)x(nx * ny) representing the  discretized
        laplacian.

    """
    n_nodes = nx * ny
    if np.isscalar(kappa):
        _kappa = np.full(n_nodes, fill_value=kappa)
    else:
        _kappa = np.array(kappa).ravel("F")

    if anisotropy_function is None:

        def _anifct(x: float, y: float) -> NDArrayFloat:
            return np.eye(2)

    else:
        _anifct = anisotropy_function

    # construct an empty sparse matrix (lil_format because it supports indexing and
    # slicing).
    lap = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

    # Looping on all nodes and considering neighbors
    for i in range(nx):
        for j in range(ny):
            # For the owner node itself:
            node_index = indices_to_node_number(i, nx, j)
            _x, _y = (i + 0.5) * dx, (j + 0.5) * dy
            lap[node_index, node_index] += (
                _kappa[node_index] ** 2 * dx * dy
                + dy
                / dx
                * (_anifct(_x + 0.5 * dx, _y)[0][0] + _anifct(_x - 0.5 * dx, _y)[0][0])
                + dx
                / dy
                * (_anifct(_x, _y + 0.5 * dy)[1][1] + _anifct(_x, _y - 0.5 * dy)[1][1])
            )

            # The four closest neighbours:
            if i > 0:
                neighbor_index = indices_to_node_number(i - 1, nx, j)
                lap[node_index, neighbor_index] -= dy / dx * _anifct(_x - 0.5 * dx, _y)[
                    0, 0
                ] - 1.0 / 4.0 * (
                    _anifct(_x, _y + 0.5 * dy)[0, 1] - _anifct(_x, _y - 0.5 * dy)[0, 1]
                )
            if i < nx - 1:
                neighbor_index = indices_to_node_number(i + 1, nx, j)
                lap[node_index, neighbor_index] -= dy / dx * _anifct(_x + 0.5 * dx, _y)[
                    0, 0
                ] + 1.0 / 4.0 * (
                    _anifct(_x, _y + 0.5 * dy)[0, 1] - _anifct(_x, _y - 0.5 * dy)[0, 1]
                )
            if j < ny - 1:
                neighbor_index = indices_to_node_number(i, nx, j + 1)
                lap[node_index, neighbor_index] -= dx / dy * _anifct(_x, _y + 0.5 * dy)[
                    1, 1
                ] + 1.0 / 4.0 * (
                    _anifct(_x + 0.5 * dx, _y)[1, 0] - _anifct(_x - 0.5 * dx, _y)[1, 0]
                )
            if j > 0:
                neighbor_index = indices_to_node_number(i, nx, j - 1)
                lap[node_index, neighbor_index] -= dx / dy * _anifct(_x, _y - 0.5 * dy)[
                    1, 1
                ] - 1.0 / 4.0 * (
                    _anifct(_x + 0.5 * dx, _y)[1, 0] - _anifct(_x - 0.5 * dx, _y)[1, 0]
                )

            # Lastly, the four diagonally closest neighbours:
            if i > 0 and j > 0:
                neighbor_index = indices_to_node_number(i - 1, nx, j - 1)
                lap[node_index, neighbor_index] -= (
                    1.0
                    / 4.0
                    * (
                        _anifct(_x, _y - 0.5 * dy)[0, 1]
                        + _anifct(_x - 0.5 * dx, _y)[1, 0]
                    )
                )
            if i < nx - 1 and j > 0:
                neighbor_index = indices_to_node_number(i + 1, nx, j - 1)
                lap[node_index, neighbor_index] += (
                    1.0
                    / 4.0
                    * (
                        _anifct(_x, _y - 0.5 * dy)[0, 1]
                        + _anifct(_x + 0.5 * dx, _y)[1, 0]
                    )
                )
            if i > 0 and j < ny - 1:
                neighbor_index = indices_to_node_number(i - 1, nx, j + 1)
                lap[node_index, neighbor_index] += (
                    1.0
                    / 4.0
                    * (
                        _anifct(_x, _y + 0.5 * dy)[0, 1]
                        + _anifct(_x - 0.5 * dx, _y)[1, 0]
                    )
                )
            if i < nx - 1 and j < ny - 1:
                neighbor_index = indices_to_node_number(i + 1, nx, j + 1)
                lap[node_index, neighbor_index] -= (
                    1.0
                    / 4.0
                    * (
                        _anifct(_x, _y + 0.5 * dy)[0, 1]
                        + _anifct(_x + 0.5 * dx, _y)[1, 0]
                    )
                )

    # Convert from lil to csr matrix for more efficient calculation
    return lap.tocsc() / dx / dy


def get_laplacian_matrix_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    kappa: Union[NDArrayFloat, float],
    anisotropy_function: Optional[Callable[[float, float, float], NDArrayFloat]] = None,
) -> csc_matrix:
    """
    Return a sparse matrix of the discretization of the Laplacian.

    See [1] for the details.

    [1]	M. O. Berild and G.-A. Fuglstad, “Spatially varying anisotropy for Gaussian
    random fields in three-dimensional space,” Spatial Statistics, vol. 55, p. 100750,
    Jun. 2023, doi: 10.1016/j.spasta.2023.100750.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz : int
        Number of meshes along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : : Union[float, NDArrayFloat]
        Range (length scale).
    anisotropy_function: Optional[Callable[[float, float, float], NDArrayFloat]]
        Anisotropy function that returns a 3d transformation value for a given
        position (x, y, z). The default is None.

    Returns
    -------
    csc_matrix
        Sparse matrix with dimension (nx * ny * nz)x(nx * ny * nz) representing the
        discretized laplacian.

    """
    n_nodes = nx * ny * nz
    if np.isscalar(kappa):
        _kappa = np.full(n_nodes, fill_value=kappa)
    else:
        _kappa = np.array(kappa).ravel("F")

    if anisotropy_function is None:

        def _anifct(x: float, y: float, z: float) -> NDArrayFloat:
            return np.eye(3)

    else:
        _anifct = anisotropy_function

    # construct an empty sparse matrix (lil_format because it supports indexing and
    # slicing).
    lap = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

    # Looping on all nodes and considering neighbors
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # For the owner node itself:
                node_index = indices_to_node_number(i, nx, j, ny, k)
                _x, _y, _z = (i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz

                lap[node_index, node_index] += (
                    _kappa[node_index] ** 2 * dx * dy * dz
                    + dy
                    * dz
                    / dx
                    * (
                        _anifct(_x + 0.5 * dx, _y, _z)[0][0]
                        + _anifct(_x - 0.5 * dx, _y, _z)[0][0]
                    )
                    + dx
                    * dz
                    / dy
                    * (
                        _anifct(_x, _y + 0.5 * dy, _z)[1][1]
                        + _anifct(_x, _y - 0.5 * dy, _z)[1][1]
                    )
                    + dx
                    * dy
                    / dz
                    * (
                        _anifct(_x, _y, _z + 0.5 * dz)[2][2]
                        + _anifct(_x, _y, _z - 0.5 * dz)[2][2]
                    )
                )

                # The six closest neighbours:
                if i < nx - 1:
                    neighbor_index = indices_to_node_number(i + 1, nx, j, ny, k)
                    lap[node_index, neighbor_index] -= (
                        dz * dy / dx * _anifct(_x + 0.5 * dx, _y, _z)[0, 0]
                        + dy
                        / 4.0
                        * (
                            _anifct(_x, _y, _z + 0.5 * dz)[0, 1]
                            - _anifct(_x, _y, _z - 0.5 * dz)[0, 1]
                        )
                        + dz
                        / 4.0
                        * (
                            _anifct(_x, _y + 0.5 * dy, _z)[0, 2]
                            - _anifct(_x, _y - 0.5 * dy, _z)[0, 2]
                        )
                    )

                if i > 0:
                    neighbor_index = indices_to_node_number(i - 1, nx, j, ny, k)
                    lap[node_index, neighbor_index] -= (
                        dz * dy / dx * _anifct(_x - 0.5 * dx, _y, _z)[0, 0]
                        - dy
                        / 4.0
                        * (
                            _anifct(_x, _y, _z + 0.5 * dz)[0, 1]
                            - _anifct(_x, _y, _z - 0.5 * dz)[0, 1]
                        )
                        - dz
                        / 4.0
                        * (
                            _anifct(_x, _y + 0.5 * dy, _z)[0, 2]
                            - _anifct(_x, _y - 0.5 * dy, _z)[0, 2]
                        )
                    )
                if j < ny - 1:
                    neighbor_index = indices_to_node_number(i, nx, j + 1, ny, k)
                    lap[node_index, neighbor_index] -= (
                        dx * dz / dy * _anifct(_x, _y + 0.5 * dy, _z)[1, 1]
                        + dx
                        / 4.0
                        * (
                            _anifct(_x, _y, _z + 0.5 * dz)[1, 2]
                            - _anifct(_x, _y, _z - 0.5 * dz)[1, 2]
                        )
                        + dz
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y, _z)[1, 0]
                            - _anifct(_x - 0.5 * dx, _y, _z)[1, 0]
                        )
                    )

                if j > 0:
                    neighbor_index = indices_to_node_number(i, nx, j - 1, ny, k)
                    lap[node_index, neighbor_index] -= (
                        dx * dz / dy * _anifct(_x, _y - 0.5 * dy, _z)[1, 1]
                        - dx
                        / 4.0
                        * (
                            _anifct(_x, _y, _z + 0.5 * dz)[1, 2]
                            - _anifct(_x, _y, _z - 0.5 * dz)[1, 2]
                        )
                        - dz
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y, _z)[1, 0]
                            - _anifct(_x - 0.5 * dx, _y, _z)[1, 0]
                        )
                    )

                if k < nz - 1:
                    neighbor_index = indices_to_node_number(i, nx, j, ny, k + 1)
                    lap[node_index, neighbor_index] -= (
                        dy * dx / dz * _anifct(_x, _y, _z + 0.5 * dz)[2, 2]
                        + dy
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y, _z)[2, 0]
                            - _anifct(_x - 0.5 * dx, _y, _z)[2, 0]
                        )
                        + dx
                        / 4.0
                        * (
                            _anifct(_x, _y + 0.5 * dy, _z)[2, 1]
                            - _anifct(_x, _y - 0.5 * dy, _z)[2, 1]
                        )
                    )

                if k > 0:
                    neighbor_index = indices_to_node_number(i, nx, j, ny, k - 1)
                    lap[node_index, neighbor_index] -= (
                        dy * dx / dz * _anifct(_x, _y, _z - 0.5 * dz)[2, 2]
                        - dy
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y, _z)[2, 0]
                            - _anifct(_x - 0.5 * dx, _y, _z)[2, 0]
                        )
                        - dx
                        / 4.0
                        * (
                            _anifct(_x, _y + 0.5 * dy, _z)[2, 1]
                            - _anifct(_x, _y - 0.5 * dy, _z)[2, 1]
                        )
                    )

                # Lastly, the twelve diagonally closest neighbours:

                # 1) i + 1, k + 1
                if i < nx - 1 and k < nz - 1:
                    neighbor_index = indices_to_node_number(i + 1, nx, j, ny, k + 1)
                    lap[node_index, neighbor_index] -= (
                        dy
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y, _z)[2, 0]
                            + _anifct(_x, _y, _z + 0.5 * dz)[0, 2]
                        )
                    )

                # 2) i - 1, k - 1
                if i > 0 and k > 0:
                    neighbor_index = indices_to_node_number(i - 1, nx, j, ny, k - 1)
                    lap[node_index, neighbor_index] -= (
                        dy
                        / 4.0
                        * (
                            _anifct(_x - 0.5 * dx, _y, _z)[2, 0]
                            + _anifct(_x, _y, _z - 0.5 * dz)[0, 2]
                        )
                    )

                # 3) i - 1, k + 1
                if i > 0 and k < nz - 1:
                    neighbor_index = indices_to_node_number(i - 1, nx, j, ny, k + 1)
                    lap[node_index, neighbor_index] += (
                        dy
                        / 4.0
                        * (
                            _anifct(_x - 0.5 * dx, _y, _z)[2, 0]
                            + _anifct(_x, _y, _z + 0.5 * dz)[0, 2]
                        )
                    )

                # 4) i + 1, k - 1
                if i < nx - 1 and k > 0:
                    neighbor_index = indices_to_node_number(i + 1, nx, j, ny, k - 1)
                    lap[node_index, neighbor_index] += (
                        dy
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y, _z)[2, 0]
                            + _anifct(_x, _y, _z - 0.5 * dz)[0, 2]
                        )
                    )

                # 5) j + 1, k + 1
                if j < ny - 1 and k < nz - 1:
                    neighbor_index = indices_to_node_number(i, nx, j + 1, ny, k + 1)
                    lap[node_index, neighbor_index] -= (
                        dx
                        / 4.0
                        * (
                            _anifct(_x, _y + 0.5 * dy, _z)[2, 1]
                            + _anifct(_x, _y, _z + 0.5 * dz)[1, 2]
                        )
                    )

                # 6) j - 1, k - 1
                if j > 0 and k > 0:
                    neighbor_index = indices_to_node_number(i, nx, j - 1, ny, k - 1)
                    lap[node_index, neighbor_index] -= (
                        dx
                        / 4.0
                        * (
                            _anifct(_x, _y - 0.5 * dy, _z)[2, 1]
                            + _anifct(_x, _y, _z - 0.5 * dz)[1, 2]
                        )
                    )

                # 7) j - 1, k + 1
                if j > 0 and k < nz - 1:
                    neighbor_index = indices_to_node_number(i, nx, j - 1, ny, k + 1)
                    lap[node_index, neighbor_index] += (
                        dx
                        / 4.0
                        * (
                            _anifct(_x, _y - 0.5 * dy, _z)[2, 1]
                            + _anifct(_x, _y, _z + 0.5 * dz)[1, 2]
                        )
                    )

                # 8) j + 1, k - 1
                if j < ny - 1 and k > 0:
                    neighbor_index = indices_to_node_number(i, nx, j + 1, ny, k - 1)
                    lap[node_index, neighbor_index] += (
                        dx
                        / 4.0
                        * (
                            _anifct(_x, _y + 0.5 * dy, _z)[2, 1]
                            + _anifct(_x, _y, _z - 0.5 * dz)[1, 2]
                        )
                    )

                # 9) i + 1, k + 1
                if i < nx - 1 and k < nz - 1:
                    neighbor_index = indices_to_node_number(i + 1, nx, j, ny, k + 1)
                    lap[node_index, neighbor_index] -= (
                        dz
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y, _z)[1, 0]
                            + _anifct(_x, _y, _z + 0.5 * dz)[0, 1]
                        )
                    )

                # 10) i - 1, j - 1
                if i > 0 and j > 0:
                    neighbor_index = indices_to_node_number(i - 1, nx, j - 1, ny, k)
                    lap[node_index, neighbor_index] -= (
                        dz
                        / 4.0
                        * (
                            _anifct(_x - 0.5 * dx, _y - 0.5 * dy, _z)[1, 0]
                            + _anifct(_x, _y - 0.5 * dy, _z)[0, 1]
                        )
                    )

                # 11) i + 1, j - 1
                if i < nx - 1 and j > 0:
                    neighbor_index = indices_to_node_number(i + 1, nx, j - 1, ny, k)
                    lap[node_index, neighbor_index] += (
                        dz
                        / 4.0
                        * (
                            _anifct(_x + 0.5 * dx, _y - 0.5 * dy, _z)[1, 0]
                            + _anifct(_x, _y - 0.5 * dy, _z)[0, 1]
                        )
                    )

                # 12) i - 1, j + 1
                if i > 0 and j < ny - 1:
                    neighbor_index = indices_to_node_number(i - 1, nx, j + 1, ny, k)
                    lap[node_index, neighbor_index] += (
                        dz
                        / 4.0
                        * (
                            _anifct(_x - 0.5 * dx, _y, _z)[1, 0]
                            + _anifct(_x, _y + 0.5 * dy, _z)[0, 1]
                        )
                    )

    # Convert from lil to csr matrix for more efficient calculation
    return lap.tocsc() / (dx * dy * dz)


def get_preconditioner(mat: csc_matrix) -> LinearOperator:
    """Get the preconditioner for the given matrix."""
    op = splu(mat)

    def super_lu(x):
        return op.solve(x)

    return LinearOperator(mat.shape, super_lu)


def get_det_H(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    anisotropy_function: Callable[[float, float, float], NDArrayFloat],
) -> NDArrayFloat:
    """
    Get the determinant of H, the anisotropy operator.

    Parameters
    ----------
    nx : int
        _description_
    ny : int
        _description_
    nz : int
        _description_
    dx : float
        _description_
    dy : float
        _description_
    dz : float
        _description_
    anisotropy_function : Callable[[float, float, float], NDArrayFloat]
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    size = nx * ny * nz
    # output array
    out = np.zeros(size)

    def get_linspace(_n, _d) -> NDArrayFloat:
        return np.linspace(0.5 * _d, (_n - 0.5) * _d, _n)

    # 3d version
    if nx != 1 and ny != 1 and nz != 1:
        X, Y, Z = np.meshgrid(
            get_linspace(nx, dx), get_linspace(ny, dy), get_linspace(nz, dz)
        )
        X = X.ravel()
        Y = Y.ravel()
        Z = Z.ravel()
        for i in range(size):
            out[i] = scipy.linalg.det(anisotropy_function(X[i], Y[i], Z[i]))
        return out

    # 2d version
    if ny != 1:
        X, Y = np.meshgrid(get_linspace(nx, dx), get_linspace(ny, dy))
        X = X.ravel()
        Y = Y.ravel()
        for i in range(size):
            out[i] = scipy.linalg.det(anisotropy_function(X[i], Y[i]))
        return out

    # 1d version
    for i in range(size):
        out[i] = scipy.linalg.det(anisotropy_function((i + 0.5) * dx))
    return out


def _get_precision_matrix(
    laplacian_matrix: csc_matrix,
    kappa: NDArrayFloat,
    alpha: int,
    mesh_size: NDArrayInt,
    mesh_dims: NDArrayFloat,
    anisotropy_function: Optional[Callable[[float, float, float], NDArrayFloat]] = None,
) -> csc_matrix:
    """
    Get the precision matrix for the given SPDE field parameters.

    Parameters
    ----------
    alpha : int
        SPDE parameter linked to the field regularity.
    spatial_dim : int
        Spatial dimension of the grid (1, 2 or 3).

    Returns
    -------
    csc_matrix
        The sparse precision matrix.
    """
    # Check if 2 alpha is an integer
    if alpha < 1.0 or not float(alpha).is_integer():
        raise ValueError(
            "alpha must be superior or equal to 1.0 and must be an whole number!"
        )

    Af = laplacian_matrix

    # if alpha > 1:
    #     for i in range(int(alpha -1)):
    #         # Af = A @ Af  # matrix multiplication
    #         Af = laplacian_matrix @ Af

    # Correction factor for variance
    nu = 2 * alpha - mesh_dims.size / 2

    if anisotropy_function is None:
        tau = (
            (kappa) ** nu
            * np.sqrt(np.math.gamma(2 * alpha))
            * (4 * np.pi) ** (mesh_dims.size / 4)
            / np.sqrt(np.math.gamma(nu))
        )
    else:
        # tau = (
        #     (_kappa) ** nu
        #     * np.sqrt(np.math.gamma(2 * alpha))
        #     * (4 * np.pi) ** (mesh_dims.size / 4)
        #     / np.sqrt(np.math.gamma(nu))
        # )
        if mesh_dims.size == 3:
            det = get_det_H(
                mesh_size[0],
                mesh_size[1],
                mesh_size[2],
                mesh_dims[0],
                mesh_dims[1],
                mesh_dims[2],
                anisotropy_function,
            )
        elif mesh_dims.size == 2:
            det = get_det_H(
                mesh_size[0],
                mesh_size[1],
                1,
                mesh_dims[0],
                mesh_dims[1],
                1.0,
                anisotropy_function,
            )
        else:
            det = get_det_H(
                mesh_size[0], 1, 1, mesh_dims[0], 1.0, 1.0, anisotropy_function
            )

        tau = (
            np.sqrt(det)
            * (kappa) ** nu
            * np.sqrt(np.math.gamma(2 * alpha))
            * (4 * np.pi) ** (2 / 4)
            / np.sqrt(np.math.gamma(nu))
        )
        tau = 1.0

        print(tau)

    # Calculate precision matrix
    Af = Af.multiply(1 / tau)
    return (Af.T @ Af) * np.prod(mesh_dims)


def get_precision_matrix_1d(
    nx: int,
    dx: float,
    kappa: NDArrayFloat,
    alpha: int,
) -> csc_matrix:
    """
    Get the precision matrix for the given SPDE field parameters.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    dx : float
        Size of the mesh along x.
    kappa : NDArrayFloat
        SPDE parameter linked to the inverse of the correlation range of the covariance
        function. Real strictly
        positive.
    alpha : int
        SPDE parameter linked to the field regularity.

    Returns
    -------
    csc_matrix
        The sparse precision matrix.
    """
    # Discretization of (kappa^2 - Delta)^(alpha)
    # Build the laplacian matrix: (kappa^2 - Delta)
    A: csc_matrix = get_laplacian_matrix_1d(nx, dx, kappa)
    return _get_precision_matrix(
        A, kappa=kappa, alpha=alpha, mesh_size=np.array([nx]), mesh_dims=np.array([dx])
    )


def get_precision_matrix_2d(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    kappa: Union[float, NDArrayFloat],
    alpha: int,
    anisotropy_function: Optional[Callable[[float, float], NDArrayFloat]] = None,
) -> csc_matrix:
    """
    Get the precision matrix for the given SPDE field parameters.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz : int
        Number of meshes along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : NDArrayFloat
        SPDE parameter linked to the inverse of the correlation range of the covariance
        function. Real strictly
        positive.
    alpha : int
        SPDE parameter linked to the field regularity.
    anisotropy_function: Optional[Callable[[float, float], NDArrayFloat]]
        Anisotropy function that returns a 2d transformation value for a given
        position (x, y). The default is None.

    Returns
    -------
    csc_matrix
        The sparse precision matrix.
    """
    # Discretization of (kappa^2 - Delta)^(alpha)
    # Build the laplacian matrix: (kappa^2 - Delta)
    A: csc_matrix = get_laplacian_matrix_2d(
        nx, ny, dx, dy, kappa, anisotropy_function=anisotropy_function
    )
    return _get_precision_matrix(
        A,
        kappa=kappa,
        alpha=alpha,
        mesh_size=np.array([nx, ny]),
        mesh_dims=np.array([dx, dy]),
        anisotropy_function=anisotropy_function,
    )


def get_precision_matrix_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    kappa: NDArrayFloat,
    alpha: int,
    anisotropy_function: Optional[Callable[[float, float, float], NDArrayFloat]] = None,
) -> csc_matrix:
    """
    Get the precision matrix for the given SPDE field parameters.

    Parameters
    ----------
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    nz : int
        Number of meshes along z.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    dz : float
        Size of the mesh along z.
    kappa : NDArrayFloat
        SPDE parameter linked to the inverse of the correlation range of the covariance
        function. Real strictly
        positive.
    alpha : int
        SPDE parameter linked to the field regularity.
    anisotropy_function: Optional[Callable[[float, float, float], NDArrayFloat]]
        Anisotropy function that returns a 3d transformation value for a given
        position (x, y, z). The default is None.

    Returns
    -------
    csc_matrix
        The sparse precision matrix.
    """
    # Discretization of (kappa^2 - Delta)^(alpha)
    # Build the laplacian matrix: (kappa^2 - Delta)
    A: csc_matrix = get_laplacian_matrix(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        kappa,
    )
    return _get_precision_matrix(
        A,
        kappa=kappa,
        alpha=alpha,
        mesh_size=np.array([nx, ny, nz]),
        mesh_dims=np.array([dx, dy, dz]),
        anisotropy_function=anisotropy_function,
    )


def simu_nc(
    cholQ: Factor,
    random_state: Optional[
        Union[int, np.random.Generator, np.random.RandomState]
    ] = None,
) -> NDArrayFloat:
    """
    Return a non conditional simulation for the given precision matrix factorization.

    Parameters
    ----------
    cholQ : Factor
        The cholesky factorization of precision matrix.
    random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used. The default is None

    Returns
    -------
    NDArrayFloat
        The non conditional simulation.

    """
    # Random state for v0 vector used by eigsh and svds
    if random_state is not None:
        random_state = check_random_state(random_state)
    else:
        random_state = np.random.default_rng()

    w = random_state.normal(size=cholQ.L().shape[0])  # white noise
    return cholQ.apply_Pt(cholQ.solve_Lt(1.0 / np.sqrt(cholQ.D()) * w))


def condition_precision_matrix(
    Q: csc_matrix, dat_indices: NDArrayInt, dat_var: NDArrayFloat
) -> csc_matrix:
    """
    Condition the precision matrix with the variance of known data points.

    Parameters
    ----------
    Q : csc_matrix
        _description_
    dat_indices : NDArrayInt
        _description_
    dat_var : NDArrayFloat
        _description_

    Returns
    -------
    csc_matrix
        The conditioned precision matrix.
    """
    # Build the diagonal matrix containing the inverse of the error variance at known
    # data points

    diag_var = lil_matrix(Q.shape)
    diag_var[dat_indices, dat_indices] = 1 / dat_var
    return (diag_var + Q).tocsc()


def kriging(
    Q: csc_matrix,
    dat: NDArrayFloat,
    dat_indices: NDArrayInt,
    cholQ: Optional[Factor] = None,
    dat_var: Optional[NDArrayFloat] = None,
) -> NDArrayFloat:
    if cholQ is None:
        _cholQ = cholesky(Q.tocsc())
    else:
        _cholQ = cholQ
    input = np.zeros(Q.shape[0])
    input[dat_indices] = dat
    if dat_var is not None:
        input[dat_indices] /= dat_var
    return _cholQ(input)


def simu_c(
    cholQ: Factor,
    Q_cond: csc_matrix,
    cholQ_cond: Factor,
    dat: NDArrayFloat,
    dat_indices: NDArrayInt,
    dat_var: NDArrayFloat,
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]],
) -> NDArrayFloat:
    """_summary_

    Parameters
    ----------
    cholQ : Factor
        _description_
    Q_cond : csc_matrix
        _description_
    cholQ_cond : Factor
        _description_
    dat : NDArrayFloat
        _description_
    dat_indices : NDArrayInt
        _description_
    dat_var : NDArrayFloat
        _description_
    random_state : Optional[Union[int, np.random.Generator, np.random.RandomState]]
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used. The default is None

    Returns
    -------
    NDArrayFloat
        _description_
    """
    z_k = kriging(Q_cond, dat, dat_indices, cholQ=cholQ_cond, dat_var=dat_var)
    # z_k = krig_prec2(Q_cond, dat * 1 / grid_var[dat_indices], dat_indices)
    z_nc = simu_nc(cholQ, random_state)
    dat_nc = z_nc[dat_indices]
    # z_nck = krig_chol(QTT_factor, QTD, dat_nc, dat_indices)
    z_nck = kriging(Q_cond, dat_nc, dat_indices, cholQ=cholQ_cond, dat_var=dat_var)
    return z_k - (z_nc - z_nck)


def sk_matern_kernel(
    x: NDArrayFloat, kappa: float, alpha: float, spatial_dim: int
) -> NDArrayFloat:
    """
    Computes Matérn correlation function for given distances (using the sklearn lib).

    Parameters:
    -----------
    x : array
        Distances between locations.
    kappa : float
        1 / range parameter (ϕ). Must be greater than 0.
    alpha : float
        Regularity parameter (nu). Must be greater than 0.
    spatial_dim: int
        Spatial dimension (1, 2 or 3).
    Returns:
    --------
    Array giving Matern correlation for given distances.
    """
    nu = 2 * alpha - spatial_dim / 2
    return skMatern(length_scale=1 / kappa, nu=nu)(0, x.ravel()[:, np.newaxis]).ravel()


def matern_kernel(
    r: NDArrayFloat, length_scale: float = 1, v: float = 1
) -> NDArrayFloat:
    """
    Computes Matérn correlation function for given distances.

    Parameters:
    -----------
    r : array
        Distances between locations.
    length_scale : float
        Range parameter (ϕ). Must be greater than 0.
    v : float
        Smoothness parameter (nu). Must be greater than 0.
    Returns:
    --------
    Array giving Matern correlation for given distances.
    """
    r = np.abs(r)
    r[r == 0] = 1e-8
    return (
        2 ** (1 - v)
        / scipy.special.gamma(v)
        * (np.sqrt(2 * v) * r / length_scale) ** v
        * scipy.special.kv(v, np.sqrt(2 * v) * r / length_scale)
    )


def get_exp_var(
    z: NDArrayFloat,
    n0: int,
    lag: float,
    nbpts: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    rng: np.random.Generator,
) -> NDArrayFloat:
    """
    Return the experimental variogram of a simulation on a grid.

    Parameters
    ----------
    z : NDArrayFloat
        Simulation vector.
    n0 : int
        Number of points sampled in the simulation.
    lag : float
        Lag of the variogram.
    nbpts : int
        Number of points in the variogram.
    nx : int
        Number of meshes along x.
    ny : int
        Number of meshes along y.
    dx : float
        Size of the mesh along x.
    dy : float
        Size of the mesh along y.
    rng: np.random.Generator
        Random number generator.

    Returns
    -------
    NDArrayFloat
        Two columns matrices. The first column contains the lag of the variogram and
        the second column the variogram.
    """
    kech = rng.choice(nx * ny, size=n0, replace=False)
    xech = ((kech - 1) % nx + 1) * dx
    yech = ((kech - 1) // nx + 1) * dy

    # transpose because npts should be the first dimension
    pts = np.array([xech, yech]).T

    dist = scipy.spatial.distance_matrix(pts, pts)
    dist = dist[np.triu_indices(n0, k=1)]
    # Note: for z, take the square distances
    diff_z = (
        scipy.spatial.distance_matrix(
            np.atleast_2d(z[kech]).T, np.atleast_2d(z[kech]).T
        )
        ** 2
    )
    diff_z = diff_z[np.triu_indices(n0, k=1)]

    breaks = np.linspace(0, lag * (nbpts - 1), nbpts)
    lagcut = np.digitize(dist, breaks)
    varexp = np.array([diff_z[lagcut == i].mean() / 2 for i in range(1, nbpts + 1)])
    return np.column_stack((lag / 2 + (np.arange(nbpts) * lag), varexp))
