# resnet pz 0-12

# check $TVM_HOME
echo "now running autotvm.py"
echo "TVM_HOME: $TVM_HOME"
which python
read -p "Press any key to continue... " -n1 -s

for i in {1..3}
do
    # yolo 0-9
    # dir="run_time${i}_yolo"
    # mkdir -p "${dir}"
    # python3 autotvm.py yolo 0 1000 2>&1 | tee autotvm_yolo_0.log
    # python3 autotvm.py yolo 1 1000 2>&1 | tee autotvm_yolo_1.log
    # python3 autotvm.py yolo 2 1000 2>&1 | tee autotvm_yolo_2.log
    # python3 autotvm.py yolo 3 1000 2>&1 | tee autotvm_yolo_3.log
    # python3 autotvm.py yolo 4 1000 2>&1 | tee autotvm_yolo_4.log
    # python3 autotvm.py yolo 5 1000 2>&1 | tee autotvm_yolo_5.log
    # python3 autotvm.py yolo 6 1000 2>&1 | tee autotvm_yolo_6.log
    # python3 autotvm.py yolo 7 1000 2>&1 | tee autotvm_yolo_7.log
    # python3 autotvm.py yolo 8 1000 2>&1 | tee autotvm_yolo_8.log
    # python3 autotvm.py yolo 9 1000 2>&1 | tee autotvm_yolo_9.log

    # mv *json *csv *log  $dir

    dir="run_time${i}"
    mkdir -p "${dir}"
    python3 autotvm_mm_cpu.py 0 1000 2>&1 | tee mm_0.log
    python3 autotvm_mm_cpu.py 1 1000 2>&1 | tee mm_1.log
    python3 autotvm_mm_cpu.py 2 1000 2>&1 | tee mm_2.log
    python3 autotvm_mm_cpu.py 3 1000 2>&1 | tee mm_3.log
    python3 autotvm_mm_cpu.py 4 1000 2>&1 | tee mm_4.log
    python3 autotvm_mm_cpu.py 5 1000 2>&1 | tee mm_5.log

    mv *json *csv *log  $dir

    # dir="run_time${i}_resnet"
    # mkdir -p "${dir}"
    # python3 autotvm.py resnet 0 1000 2>&1 | tee autotvm_resnet_0.log
    # python3 autotvm.py resnet 1 1000 2>&1 | tee autotvm_resnet_1.log
    # python3 autotvm.py resnet 2 1000 2>&1 | tee autotvm_resnet_2.log
    # python3 autotvm.py resnet 3 1000 2>&1 | tee autotvm_resnet_3.log
    # python3 autotvm.py resnet 4 1000 2>&1 | tee autotvm_resnet_4.log
    # python3 autotvm.py resnet 5 1000 2>&1 | tee autotvm_resnet_5.log
    # python3 autotvm.py resnet 6 1000 2>&1 | tee autotvm_resnet_6.log
    # python3 autotvm.py resnet 7 1000 2>&1 | tee autotvm_resnet_7.log
    # python3 autotvm.py resnet 8 1000 2>&1 | tee autotvm_resnet_8.log
    # python3 autotvm.py resnet 9 1000 2>&1 | tee autotvm_resnet_9.log
    # python3 autotvm.py resnet 10 1000 2>&1 | tee autotvm_resnet_10.log
    # python3 autotvm.py resnet 11 1000 2>&1 | tee autotvm_resnet_11.log
    # python3 autotvm.py resnet 12 1000 2>&1 | tee autotvm_resnet_12.log

    # mv *json *csv *log  $dir
done