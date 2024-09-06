import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import time
import os
from copy import deepcopy
import tvm
from tvm.auto_scheduler.measure import local_builder_build, local_run
import tvm.auto_scheduler
from tvm import auto_scheduler
from tvm.auto_scheduler.measure import MeasureResult
        
@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

def run(
    task,
    input_file,
    output_file,
    timeout=20,
    verbose=0,
    number=3,
    repeat=3,
    min_repeat_ms=0,
    cooldown_interval=0,
    enable_cpu_cache_flush=True,
    dev=0,
):
    """Execute a log file and save"""
    print("input_file", input_file)
    print("output_file", output_file)
    
    readlines, _ = tvm.auto_scheduler.RecordReader(input_file).read_lines()
    
    inputs = []
    for i in range(len(readlines)):
        state = task.compute_dag.infer_bound_from_state(readlines[i].state)
        inp = tvm.auto_scheduler.MeasureInput(task, state)
        inputs.append(inp)
        
    build_results = local_builder_build(inputs, timeout, os.cpu_count(), "default", verbose)
        
    res = local_run(
        inputs,
        build_results,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
        dev,
    )
    # tvm.auto_scheduler._ffi_api.SaveRecords(final_log, inp, res)
    # tvm.auto_scheduler._ffi_api.SaveRecords("dump.json", inp, res)
    from tvm.auto_scheduler.measure_record import save_records
    save_records(output_file, inputs, res)
    
    print("done measuring for input.json")
    
    return res

import json

def extract_values_from_json(line):
    data = json.loads(line)
    value_str = data['i'][0][0]
    value_list = json.loads(value_str)
    pz = value_list[1:]
    return pz

def test_conv2d():
        
    size = [1, 17, 17, 512, 1024, 1, 1, (1, 1), (0, 0)]
    N, H, W, CO, CI, KH, KW, strides, padding = size

    target = tvm.target.cuda()
    task = auto_scheduler.SearchTask(
        func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,
    )

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    full_space_input_dir = "./"
    input_file = os.path.join(full_space_input_dir, f"test_db_sample.json")
    output_file = os.path.join(full_space_input_dir, f"test_db_sample_output.json")
    
    if not os.path.exists(input_file):
        raise ValueError("input file does not exist")
    else:
        print("input file exists, running...")
        run(task, input_file, output_file)

if __name__ == '__main__':
    test_conv2d()