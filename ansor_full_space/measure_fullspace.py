import os
import numpy as np
import tvm
from tvm import te, auto_scheduler
import time
import os
from copy import deepcopy
import tvm
from tvm.auto_scheduler.measure import local_builder_build, local_run
import tvm.auto_scheduler
from tvm import auto_scheduler
from tvm.auto_scheduler.measure import MeasureResult


# MLP bert lart/ basic
sizes=[
    #Bert large
[512,64,1024],      #BMATmul
[512,4096,1024],    #MLP1
[512,1024,4096],    #MLP2

    #Bert basic
[512,64,768],       #BMATmul
[512,3072,768],     #MLP1
[512,768,3072],     #MLP2
]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def _matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    
    return [A, B, matmul]

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
        # print("res", res)
        # print("inp", inp)
        # input('pause')
        # inputs.append(inp[0])
        # results.append(MeasureResultSpace(res))
        # results_dump.append(MeasureResult(res))
    # print("size of inputs", len(inputs))
    # print("inputs", inputs)
    # print("size of results", len(results))
    # print("results", results)
    # input('pause')
    # tvm.auto_scheduler._ffi_api.SaveRecords("dump.json", inputs, results_dump)
    
    input("done measuring for input.json")
    
    return inputs, results

def test_matmul():
        
    for i, size in enumerate(sizes):
        M=size[0]
        N=size[1]
        L=size[2]
        print("M=",M,"N=",N,"K=",L)
        target = tvm.target.Target("llvm -mcpu=core-avx2")
        task = tvm.auto_scheduler.SearchTask(func=_matmul, args=(M, L, N, "float32"), target=target)

        # Inspect the computational graph
        print("Computational DAG:")
        print(task.compute_dag)

        full_space_input_dir = "cpu_mm_full_space"
        input_file = os.path.join(full_space_input_dir, f"cpu_mm_testcase_{i}_full_space.json")
        output_file = os.path.join(full_space_input_dir, f"cpu_mm_testcase_{i}_full_space_output.json")
        
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=5,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(output_file)],
            verbose=2,
        )
        
        if not os.path.exists(input_file):
            raise ValueError("input file does not exist")
        else:
            print("input file exists, running...")
            run(task, input_file, output_file)
        
        # # Run auto-tuning (search)
        # task.tune(tune_option)
        # # Apply the best schedule
        # sch, args = task.apply_best(log_file)
        
        # func = tvm.build(sch, args, target)
        # a_np = np.random.uniform(size=(M, L)).astype(np.float32)
        # b_np = np.random.uniform(size=(L, N)).astype(np.float32)
        # out_np = a_np.dot(b_np)

        # dev = tvm.cpu()
        # a_tvm = tvm.nd.array(a_np, device=dev)
        # b_tvm = tvm.nd.array(b_np, device=dev)
        # out_tvm = tvm.nd.empty(out_np.shape, device=dev)
        # func(a_tvm, b_tvm, out_tvm)

        # # Check results
        # np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

        # # Evaluate execution time.
        # evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        # print(
        # "Execution time of this operator: %.3f ms"
        # % (np.median(evaluator(a_tvm, b_tvm, out_tvm).results) * 1000)
        # )
        # print("for M=",M,"N=",N,"K=",L,"matmul correctness check passed!")

if __name__ == '__main__':
    test_matmul()