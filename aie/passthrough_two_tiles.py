import sys
import time
import numpy as np

import aie.iron as iron
from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer

@iron.jit(is_placed=False)
def passthrough_single(input0, output):
    data_size = output.numel()
    element_type = output.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    of_in = ObjectFifo(data_ty, name="in")
    of_out = ObjectFifo(data_ty, name="out")

    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] = elem_in[i]
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a_in, c_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    program = Program(iron.get_current_device(), rt)
    return program.resolve_program(SequentialPlacer())

@iron.jit(is_placed=False)
def passthrough_twohop(input0, output):
    data_size = output.numel()
    element_type = output.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # Three ObjectFifos: in -> mid -> out
    of_in = ObjectFifo(data_ty, name="in")
    of_mid = ObjectFifo(data_ty, name="mid")
    of_out = ObjectFifo(data_ty, name="out")

    # Core 0: DDR → core0 → mid FIFO
    def core_fn0(of_in, of_mid):
        elem_in = of_in.acquire(1)
        elem_mid = of_mid.acquire(1)
        for i in range_(data_size):
            elem_mid[i] = elem_in[i]
        of_in.release(1)
        of_mid.release(1)

    # Core 1: mid FIFO → core1 → DDR
    def core_fn1(of_mid, of_out):
        elem_mid = of_mid.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] = elem_mid[i]
        of_mid.release(1)
        of_out.release(1)

    worker0 = Worker(core_fn0, [of_in.cons(), of_mid.prod()])
    worker1 = Worker(core_fn1, [of_mid.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a_in, c_out):
        rt.start(worker0, worker1)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    program = Program(iron.get_current_device(), rt)
    return program.resolve_program(SequentialPlacer())

def measure_kernel(kernel, input_tensor, output_tensor, n_iters=1000):
    # Warmup (pays JIT + CO load cost)
    kernel(input_tensor, output_tensor)

    start = time.perf_counter()
    for _ in range(n_iters):
        kernel(input_tensor, output_tensor)
    end = time.perf_counter()

    return (end - start) / n_iters

def main():
    data_size = 48
    element_type = np.int32

    # bytes moved (read + write)
    bytes_per_elem = np.dtype(element_type).itemsize
    bytes_moved = 2 * data_size * bytes_per_elem

    # Allocate NPU tensors
    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    out_single = iron.zeros_like(input0)
    out_twohop = iron.zeros_like(input0)

    passthrough_single(input0, out_single)
    if not np.all(input0.numpy() == out_single.numpy()):
        print("ERROR: Single-tile passthrough incorrect.")
        sys.exit(1)
    print("Single-tile: PASS")

    passthrough_twohop(input0, out_twohop)
    if not np.all(input0.numpy() == out_twohop.numpy()):
        print("ERROR: Two-tile passthrough incorrect.")
        sys.exit(1)
    print("Two-tile:  PASS")

    N_ITERS = 1000

    avg_single = measure_kernel(passthrough_single, input0, out_single, N_ITERS)
    avg_twohop = measure_kernel(passthrough_twohop, input0, out_twohop, N_ITERS)

    gbps_single = bytes_moved / (avg_single * 1e9)
    gbps_twohop = bytes_moved / (avg_twohop * 1e9)

    print("\n=== Performance Results ===")
    print(f"Data size      : {data_size} elements")
    print(f"Element type   : {element_type}")
    print(f"Bytes moved    : {bytes_moved} bytes (read+write)\n")

    print(f"Single-tile: runtime = {avg_single*1e6:.2f} us, "
          f"throughput = {gbps_single:.4f} GB/s")
    print(f"Two-tile:    runtime = {avg_twohop*1e6:.2f} us, "
          f"throughput = {gbps_twohop:.4f} GB/s")

    print(f"\nSlowdown (two-tile / single-tile) = {avg_twohop / avg_single:.3f}x")

if __name__ == "__main__":
    main()