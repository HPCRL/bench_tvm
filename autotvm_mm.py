import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
import time
import csv
import json
import threading
import logging
import sys
import numpy as np

import tvm
from tvm import te, topi, testing
import tvm.testing

from tvm import autotvm

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


@autotvm.template("tutorial/matmul")
def matmul(N, L, M, dtype):
    ##### define space begin #####
    cfg = autotvm.get_config()

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)
    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")


    y, x = s[C].op.axis
    (k,) = s[CC].op.reduce_axis

    cfg.define_split("tile_y", y, num_outputs=3)
    cfg.define_split("tile_x", x, num_outputs=3)
    cfg.define_split("tile_k", k, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        # llvm-based backends cannot do non-explicit unrolling
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])


    by, ty, yi = cfg["tile_y"].apply(s, C, y)
    bx, tx, xi = cfg["tile_x"].apply(s, C, x)

    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    s[C].reorder(by, bx, ty, tx, yi, xi)
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[C].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

    s[CC].compute_at(s[C], tx)
    yi, xi = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, yi, xi)
    s[CC].pragma(ki, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[CC].pragma(ki, "unroll_explicit", cfg["unroll_explicit"].val)

    s[AA].compute_at(s[CC], ko)
    s[AL].compute_at(s[CC], ki)
    s[BB].compute_at(s[CC], ko)
    s[BL].compute_at(s[CC], ki)
    y, k = s[AA].op.axis
    ty, yi = s[AA].split(y, nparts=cfg["tile_y"].size[1])
    tx, ki = s[AA].split(k, nparts=cfg["tile_x"].size[1])
    s[AA].reorder(ty, tx, yi, ki)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[AA].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

    x, k = s[BB].op.axis
    ty, xi = s[BB].split(x, nparts=cfg["tile_y"].size[1])
    tx, ki = s[BB].split(k, nparts=cfg["tile_x"].size[1])
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)
    s[BB].reorder(ty, tx, xi, ki)
    s[BB].pragma(xi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[BB].pragma(xi, "unroll_explicit", cfg["unroll_explicit"].val)
    return s, [A, B, C]

def parse_and_write(log_file_path, csv_file_path, start_time):
    if not os.path.exists(log_file_path):
        log_file_path = "x" + log_file_path
    # deal with FileNotFoundError exception
    try:
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
    except FileNotFoundError:
        # print(f'Json log might not be generated yet. Skip this round.', flush=True)
        return

    lines = log_content.splitlines()
    num_lines = len(lines)
    min_time_value = float('inf')

    for line in lines:
        try:
            data = json.loads(line)
            time_value = data['result'][0][0]

            if time_value < min_time_value:
                min_time_value = time_value
        except json.JSONDecodeError:
            continue

    if min_time_value != float('inf'):
        elapsed_time = int(time.time() - start_time)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([elapsed_time, min_time_value, num_lines])


timer = None

def start_timer(interval, log_file_path, csv_file_path, start_time):
    global timer
    timer = threading.Timer(interval, start_timer, [interval, log_file_path, csv_file_path, start_time])
    timer.start()
    parse_and_write(log_file_path, csv_file_path, start_time)
    
def stop_timer():
    global timer
    if timer:
        timer.cancel()
        
def caller_autotvm(specify_pz, ntrials):
    # logging config (for printing tuning log to screen)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    print("specify_pz: ", specify_pz, flush=True)
    print("ntrials: ", ntrials, flush=True)
    
    # if we have specify_pz, we only test that case
    if specify_pz != "-1":
        print("testing specified case: ", specify_pz, flush=True)
        sizes_tmp = [sizes[int(specify_pz)]]
    else:
        print("testing all cases", flush=True)
        sizes_tmp = sizes
        
    for i, size in enumerate(sizes):
        M=size[0]
        N=size[1]
        L=size[2]
        if size not in sizes_tmp:
            continue
        print("M=",M,"N=",N,"K=",L, flush=True)
        task = autotvm.task.create("tutorial/matmul", args=(M, L, N, "float32"), target="cuda")
        
        start_time = time.time()
        # Use the AutoTVM tuner, for example, GridSearchTuner
        tuner = autotvm.tuner.XGBTuner(task)
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
        )

        log_file = "cuda_testCase_" + str(i) +"_matmul_M"+str(M)+"_N"+str(N)+"_K"+str(L)+".json"

        start_time = int(time.time())
        csv_file_path = log_file.replace('.json', '.csv')
        
        # write the start time to the csv file
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_file.write(f"start_time:{str(start_time)}\n")

        tuner.tune(
            n_trial=ntrials,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(log_file)],
        )

        end_time = time.time()
        search_time = end_time - start_time
        search_time /= 60
        
        # wait 3 seconds for the last log to be written
        time.sleep(3)
        # stop_timer()
        print(f"Total search time: {search_time} minutes", flush=True)

        dispatch_context = autotvm.apply_history_best(log_file)
        best_config = dispatch_context.query(task.target, task.workload)
        print("\nBest config:")
        print(best_config)

        # apply history best from log file
        with autotvm.apply_history_best(log_file):
            with tvm.target.Target("cuda"):
                s, arg_bufs = matmul(N, L, M, "float32")
                func = tvm.build(s, arg_bufs)

        # check correctness
        a_np = np.random.uniform(size=(N, L)).astype(np.float32)
        w_np = np.random.uniform(size=(L, M)).astype(np.float32)
        c_np = np.matmul(a_np, w_np)

        dev = tvm.cuda()
        a_tvm = tvm.nd.array(a_np, device=dev)
        w_tvm = tvm.nd.array(w_np, device=dev)
        c_tvm = tvm.nd.empty(c_np.shape, device=dev)
        func(a_tvm, w_tvm, c_tvm)

        tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-2)


if __name__ == '__main__':
    import sys, os

    # Parsing command-line arguments
    specify_pz = sys.argv[1]
    ntrials = int(sys.argv[2])

    # Calling the AutoTVM tuning function
    caller_autotvm(specify_pz, ntrials)

