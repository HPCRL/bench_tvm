import logging
import sys
import numpy as np
import os
import csv
import time
import json
import threading


import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

from tvm import autotvm

sizesVGG = [
    [1, 224, 224, 64, 3, 3, 3, 1, 1],   # VGG1
    [1, 112, 112, 128, 128, 3, 3, 1, 1], # VGG3
    [1, 56, 56, 256, 256, 3, 3, 1, 1],   # VGG5
    [1, 28, 28, 512, 512, 3, 3, 1, 1],   # VGG7
    [1, 14, 14, 512, 512, 3, 3, 1, 1],   # VGG9
]

########### all pz 
sizesResnet = [
    [1, 224, 224, 64, 3, 7, 7, 2, 3],   # RESNET1
    [1, 56, 56, 64, 64, 1, 1, 1, 0],    # RESNET2
    [1, 56, 56, 64, 64, 3, 3, 1, 1],    # RESNET2
    [1, 56, 56, 256, 64, 1, 1, 1, 0],   # RESNET2
    [1, 56, 56, 128, 256, 1, 1, 2, 0],  # RESNET3
    [1, 28, 28, 128, 128, 3, 3, 1, 1],  # RESNET3
    [1, 28, 28, 512, 128, 1, 1, 1, 0],  # RESNET3
    [1, 28, 28, 256, 512, 1, 1, 2, 0],  # RESNET4
    [1, 14, 14, 256, 256, 3, 3, 1, 1],  # RESNET4
    [1, 14, 14, 1024, 256, 1, 1, 1, 0], # RESNET4
    [1, 14, 14, 512, 1024, 1, 1, 2, 0], # RESNET5
    [1, 7, 7, 512, 512, 3, 3, 1, 1],    # RESNET5
    [1, 7, 7, 2048, 512, 1, 1, 1, 0],   # RESNET5
]

sizesYolo = [
    [1, 544, 544, 32, 3, 3, 3, 1, 1],    # Yolo0
    [1, 272, 272, 64, 32, 3, 3, 1, 1],   # Yolo2
    [1, 136, 136, 128, 64, 3, 3, 1, 1],  # yolo4
    [1, 136, 136, 64, 128, 1, 1, 1, 0],  # yolo5
    [1, 68, 68, 256, 128, 3, 3, 1, 1],   # yolo8
    [1, 68, 68, 128, 256, 1, 1, 1, 0],   # yolo9
    [1, 34, 34, 512, 256, 3, 3, 1, 1],   # yolo12
    [1, 34, 34, 256, 512, 1, 1, 1, 0],   # yolo13
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],  # yolo18
    [1, 17, 17, 512, 1024, 1, 1, 1, 0],  # yolo19
]
########### all pz end

class Conv2DParams:
    def __init__(self, N, H, W, CO, CI, KH, KW, strides, padding):
        self.N = N
        self.H = H
        self.W = W
        self.CO = CO
        self.CI = CI
        self.KH = KH
        self.KW = KW
        self.strides = strides
        self.padding = padding

@autotvm.template("tutorial/conv2d_no_batching")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, "local")

    # create cache stage
    AA = s.cache_read(data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi = cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxm, rxi = cfg["tile_ry"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # tune unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    return s, [raw_data, kernel, conv]

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
        
def caller_autotvm(network, specify_pz, ntrials):
    # logging config (for printing tuning log to screen)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    print("network: ", network, flush=True)
    print("specify_pz: ", specify_pz, flush=True)
    print("ntrials: ", ntrials, flush=True)
    
    if network == "yolo":
        sizes=sizesYolo
        print(f"\ntesting yolo with {len(sizes)} layers\n")
    elif network == "resnet":
        sizes=sizesResnet
        print(f"\ntesting resnet with {len(sizes)} layers\n")
    elif network == "vgg":
        sizes=sizesVGG
        print(f"\ntesting vgg with {len(sizes)} layers\n")
    else:
        raise Exception("network not specified!")


    # if we have specify_pz, we only test that case
    if specify_pz != "-1":
        print("testing specified case: ", specify_pz, flush=True)
        if network == "yolo":
            sizes_tmp = [sizesYolo[int(specify_pz)]]
        elif network == "resnet":
            sizes_tmp = [sizesResnet[int(specify_pz)]]
        elif network == "vgg":
            sizes_tmp = [sizesVGG[int(specify_pz)]]
        else:
            raise Exception("network not specified!")
    # otherwise, we test all cases
    else:
        print("network not specified, testing all cases!", flush=True)
        sizes_tmp = sizes

    conv_params = {}
    for i, size in enumerate(sizes):
        if size not in sizes_tmp:
            continue
        N, H, W, CO, CI, KH, KW, stride, pad = size
        key = "conv" + str(i+1)
        #N, H, W, CO, CI, KH, KW, strides, padding
        conv_params[key] = Conv2DParams(N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad))


    for ite, key in enumerate(conv_params.keys()):
        if specify_pz != "-1":
            ite = int(specify_pz)
            
        start_time = time.time()
        conv = conv_params[key]
        target = tvm.target.cuda()
        
        # Use the conv2d layer to test
        N, H, W, CO, CI, KH, KW, strides, padding = conv.N, conv.H, conv.W, conv.CO, conv.CI, conv.KH, conv.KW, conv.strides, conv.padding
                
        print ("Running for: ", N, H, W, CO, CI, KH, KW, strides, padding)
        
        task = autotvm.task.create(
            "tutorial/conv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target="cuda"
        )
        
        start_time = time.time()
        # Use the AutoTVM tuner, for example, GridSearchTuner
        tuner = autotvm.tuner.XGBTuner(task)
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
        )

        log_file = "cuda_"+network+"_testCase_"+str(ite)+"_conv2d_N_"+str(N)+"_H_"+str(H)+"_W_"+str(W)+"_CO_"+str(CO)+"_CI_"+str(CI)+"_KH_"+str(KH)+"_KW_"+str(KW)+"_strides_"+str(strides)+"_padding_"+str(padding)+".json"

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

        print(task.config_space)
        
        # check correctness
        dispatch_context = autotvm.apply_history_best(log_file)
        best_config = dispatch_context.query(task.target, task.workload)
        print("\nBest config:")
        print(best_config)

        # apply history best from log file
        with autotvm.apply_history_best(log_file):
            with tvm.target.Target("cuda"):
                s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
                func = tvm.build(s, arg_bufs)

        # check correctness
        a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
        w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
        c_np = conv2d_nchw_python(a_np, w_np, strides, padding)
        

        dev = tvm.cuda()
        a_tvm = tvm.nd.array(a_np, device=dev)
        w_tvm = tvm.nd.array(w_np, device=dev)
        c_tvm = tvm.nd.empty(c_np.shape, device=dev)
        func(a_tvm, w_tvm, c_tvm)

        tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-2)

        # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
        # and the overhead of kernel launch. You can also use nvprof to validate the result.
        evaluator = func.time_evaluator(func.entry_name, dev, number=5)
        print("Time cost of this operator(xgboost): %f" % evaluator(a_tvm, w_tvm, c_tvm).mean)


if __name__ == '__main__':
    import sys, os

    # Parsing command-line arguments
    network = sys.argv[1]
    specify_pz = sys.argv[2]
    ntrials = int(sys.argv[3])

    # Calling the AutoTVM tuning function
    caller_autotvm(network, specify_pz, ntrials)

