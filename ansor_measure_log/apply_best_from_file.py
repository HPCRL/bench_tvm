import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.auto_scheduler.measure_record import load_record_from_string
import pandas as pd

@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


if __name__ == "__main__":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 17, 17, 512, 1024, 1, 1, [1, 1], [0, 0]
    
    
    target = tvm.target.Target("cuda -arch=sm_89")
    task = auto_scheduler.SearchTask(
        func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
    )
    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "test.json"
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    
    func = tvm.build(sch, args, target)
    from tvm.topi.testing import conv2d_nchw_python

    # Check correctness
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
    out_np = np.maximum(conv_np, 0.0)

    dev = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(data_tvm, weight_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
    
    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000), flush=True)
