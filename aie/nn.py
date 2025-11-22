import sys, os, time
import numpy as np
import aie.iron as iron
from aie.iron import ExternalFunction, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.config import cxx_header_path

@iron.jit(is_placed=False)
def chain_two_tiles(X, W1, W2, Y2_out):
    m = k = n = 64
    r, s, t = 2, 8, 8
    dt = Y2_out.dtype

    # tensor types
    A_ty  = np.ndarray[(m, k), np.dtype[dt]]   # X
    W1_ty = np.ndarray[(k, n), np.dtype[dt]]   # W1
    Y1_ty = np.ndarray[(m, n), np.dtype[dt]]   # Y1 (A->B bridge)
    W2_ty = np.ndarray[(n, n), np.dtype[dt]]   # W2
    Y2_ty = np.ndarray[(m, n), np.dtype[dt]]   # Y2

    of_X_DDR = iron.ObjectFifo(A_ty, name="X_DDR")
    tap_X_to_A = TensorTiler2D.group_tiler(
        (m, k),          # whole X tile
        (r, s),          # micro-tile for mmul
        (m // r, k // s) # number of micro-tiles
    )[0]
    of_X_to_A = of_X_DDR.cons().forward(
        dims_to_stream=tap_X_to_A.transformation_dims,
        name="X_to_A",
    )

    of_W1_DDR = iron.ObjectFifo(W1_ty, name="W1_DDR")
    tap_W1_to_A = TensorTiler2D.group_tiler(
        (k, n),
        (s, t),
        (k // s, n // t),
    )[0]
    of_W1_to_A = of_W1_DDR.cons().forward(
        dims_to_stream=tap_W1_to_A.transformation_dims,
        name="W1_to_A",
    )

    of_Y1_A_to_B = iron.ObjectFifo(Y1_ty, name="Y1_A_to_B")

    of_W2_DDR = iron.ObjectFifo(W2_ty, name="W2_DDR")
    tap_W2_to_B = TensorTiler2D.group_tiler(
        (n, n),
        (s, t),
        (n // s, n // t),
    )[0]
    of_W2_to_B = of_W2_DDR.cons().forward(
        dims_to_stream=tap_W2_to_B.transformation_dims,
        name="W2_to_B",
    )

    of_Y2_B = iron.ObjectFifo(Y2_ty, name="Y2_B")
    tap_Y2_to_DDR = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    of_Y2_to_DDR = of_Y2_B.cons().forward(
        dims_to_stream=tap_Y2_to_DDR.transformation_dims,
        name="Y2_to_DDR",
    )

    # kernel used by both tiles (dense layer with ReLU)
    dense = ExternalFunction(
        "dense",
        source_file=os.path.join(os.path.dirname(__file__), "dense.cc"),
        arg_types=[A_ty, W1_ty, Y1_ty],  # all 64x64 int8, reused for both layers
        include_dirs=[cxx_header_path()],
    )

    def mm(of_a, of_b, of_c, fn):  # one m×n tile
        c = of_c.acquire(1)
        # zero C before accumulation
        for i in range_(m):
            for j in range_(n):
                c[i, j] = 0
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        fn(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    # tiles
    tile_A = Worker(
        mm,
        [of_X_to_A.cons(), of_W1_to_A.cons(), of_Y1_A_to_B.prod(), dense],
    )  # Y1 = ReLU(X @ W1)

    tile_B = Worker(
        mm,
        [of_Y1_A_to_B.cons(), of_W2_to_B.cons(), of_Y2_B.prod(), dense],
    )  # Y2 = ReLU(Y1 @ W2)

    # runtime: DDR→A, DDR→B, B→DDR
    tap_A_DDR  = TensorTiler2D.group_tiler((m, k), (m, k), (1, 1))[0]
    tap_W1_DDR = TensorTiler2D.group_tiler((k, n), (k, n), (1, 1))[0]
    tap_W2_DDR = TensorTiler2D.group_tiler((n, n), (n, n), (1, 1))[0]
    tap_Y2_DDR = TensorTiler2D.group_tiler((m, n), (m, n), (1, 1))[0]

    rt = Runtime()
    with rt.sequence(A_ty, W1_ty, W2_ty, Y2_ty) as (A_buf, W1_buf, W2_buf, Y2_buf):
        rt.start(tile_A, tile_B)
        tg = rt.task_group()
        # DDR → A tile
        rt.fill(of_X_DDR.prod(),  A_buf,  tap=tap_A_DDR,  task_group=tg)
        rt.fill(of_W1_DDR.prod(), W1_buf, tap=tap_W1_DDR, task_group=tg)
        # DDR → B tile
        rt.fill(of_W2_DDR.prod(), W2_buf, tap=tap_W2_DDR, task_group=tg)
        # B → DDR (final Y2)
        rt.drain(of_Y2_to_DDR.cons(), Y2_buf,
                 tap=tap_Y2_DDR, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

def main():
    m = k = n = 64
    dtype = np.int8

    X  = iron.randint(0, 128, (m, k), dtype=dtype, device="npu")
    W1 = iron.randint(0, 128, (k, n), dtype=dtype, device="npu")
    W2 = iron.randint(0, 128, (n, n), dtype=dtype, device="npu")
    Y2 = iron.zeros(m * n, dtype=dtype, device="npu")

    # reference
    dense_1 = X.numpy() @ W1.numpy()
    relu_1  = np.maximum(dense_1, 0)
    dense_2 = relu_1 @ W2.numpy()
    relu_2  = np.maximum(dense_2, 0)
    ref = relu_2

    # run hardware version
    chain_two_tiles(X, W1, W2, Y2)
    got = Y2.numpy().reshape(m, n)

    # only print on failure
    bad = (ref != got)
    errors = int(bad.sum())
    if errors:
        idx = np.argwhere(bad)[:32]
        print(f"FAILED — mismatches: {errors}/{m*n}")
        for r, c in idx:
            print(f"  (r={r}, c={c}) ref={int(ref[r,c])} got={int(got[r,c])}")
        sys.exit(1)

    print("\nPASSED\n")

    N_ITERS = 200
    start = time.perf_counter()
    for _ in range(N_ITERS):
        chain_two_tiles(X, W1, W2, Y2)
    end = time.perf_counter()
    avg_s = (end - start) / N_ITERS

    # DDR traffic per inference (approx):
    # X (m×k) + W1 (k×n) + W2 (n×n) + Y2 (m×n)
    bytes_per_elem = np.dtype(dtype).itemsize
    bytes_moved = (m * k + k * n + n * n + m * n) * bytes_per_elem
    gbps = bytes_moved / (avg_s * 1e9)

    print(f"Average runtime (2-layer NN) = {avg_s*1e6:.2f} us")
    print(f"Approx DDR throughput        = {gbps:.4f} GB/s")

    sys.exit(0)

if __name__ == "__main__":
    main()