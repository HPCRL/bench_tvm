#!/usr/bin/python

import os
import sys
import warnings
from collections import OrderedDict

# from hypermapper import optimizer  # noqa

from subprocess import Popen, PIPE

import logging
import sys
import numpy as np
import math

import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

from tvm import autotvm


def batched_tvm_measure(tuner, measure_option, record_file, configs):
    inputs, results = tuner.batch_fake_tune(
        measure_option,
        configs,
        callbacks=[autotvm.callback.log_to_file(record_file)],
    )

def tvm_config_space(tuner, measure_option, record_file, 
                y1, y2, M,
                x1, x2, N,
                k1, K,
                max_unroll,
                explicit_unroll):
    conf_dict = OrderedDict()
    conf_dict['tile_y'] = [-1, y1, y2]
    conf_dict['tile_x'] = [-1, x1, x2]
    conf_dict['tile_k'] = [-1, k1]
    conf_dict['auto_unroll_max_step'] = max_unroll
    conf_dict['unroll_explicit'] = explicit_unroll

    if int(np.prod([y1, y2])) > M or \
       int(np.prod([x1, x2])) > N or \
       k1 > K or \
       y2 == 1 or \
       x2 == 1 or \
       y1*y2 != x1*x2:
        return
    else:
        return conf_dict

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
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
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

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_factors(x):
    factor_list = []
    for i in range(1, x + 1):
        if x % i == 0:
            factor_list.append(i)
    return factor_list

def launch_hyper(M, N, K, fi, network):
    task = autotvm.task.create("tutorial/matmul", args=(M, K, N, "float32"), target="cuda")
    
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
    )


    tuner = autotvm.tuner.BACOTuner(task)
    record_file = f"{network}_"+str(fi)+".log"
    
    
    factors_M = get_factors(M)
    factors_N = get_factors(N)
    factors_K = get_factors(K)
        
    configs = []
    for y1 in factors_M:
        for y2 in factors_M:
            for x1 in factors_N:
                for x2 in factors_N:
                    for k1 in factors_K:
                        for max_unroll in [0, 512, 1500]:
                            for explicit_unroll in [0, 1]:
                                conf_dict = tvm_config_space(
                                    tuner, measure_option, record_file,
                                    y1, y2, M,
                                    x1, x2, N,
                                    k1, K,
                                    max_unroll,
                                    explicit_unroll
                                )
                                if conf_dict:
                                    configs.append(conf_dict)

    # size of configs
    print(f"M = {M}, N = {N}, K = {K}, sizes of configs: {len(configs)}")
    
    batched_tvm_measure(tuner, measure_option, record_file, configs)
                        
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

if __name__ == "__main__":
    
    import sys
    network = 'mm'
    for fi, size in enumerate(sizes):
        M, N, K = size
        launch_hyper(M, N, K, fi, network)
        # print(f"End of {network} {fi}")