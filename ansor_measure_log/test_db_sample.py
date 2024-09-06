
import tempfile

import numpy as np

import tvm
from tvm import auto_scheduler

from tvm.testing.auto_scheduler import matmul_auto_scheduler_test
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


def get_sample_records(log_file, number, task):
    """
    Generate a list of random MeasureInput and MeasureResult pairs.

    Args:
        log_file: The path to the log file where the records will be saved.
        number: The number of random MeasureInput and MeasureResult pairs to generate.
        task: The task for which the MeasureInput and MeasureResult pairs will be generated.

    Returns:
        tuple: A tuple containing the task, the list of MeasureInput objects
                and the list of MeasureResult objects.
    """
    print("===================================", flush=True)
    print(">>>>  Sampling Init Population <<<<", flush=True)
    print("===================================", flush=True)

    runner = auto_scheduler.LocalRunner(timeout=10)
    builder = auto_scheduler.LocalBuilder()
    
    policy = auto_scheduler.SketchPolicy(
        task, verbose=0,
    )
    # policy = auto_scheduler.SketchPolicy(task, program_cost_model=RandomModel(), verbose=0)
    states = policy.sample_initial_population()
    states = states[: min(number, len(states))]

    inputs = [auto_scheduler.MeasureInput(task, s) for s in states]

    bress = builder.build(inputs)
    mress = runner.run(inputs, bress)

    with open(log_file, "a") as file:
        auto_scheduler.save_records(file.name, inputs, mress)

    return inputs, mress

size = [1, 17, 17, 512, 1024, 1, 1, (1, 1), (0, 0)]
N, H, W, CO, CI, KH, KW, strides, padding = size

target = tvm.target.cuda()
task = auto_scheduler.SearchTask(
    func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,
)

log_file = "test_db_sample.json"

inputs, results = get_sample_records(log_file, 200, task)