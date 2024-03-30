# resnet pz 0-12

# check $TVM_HOME
echo "TVM_HOME: $TVM_HOME"

# check
echo "checking...."
echo "now running autotvm.py"
echo "resnet 0-12"
read -p "Press any key to continue... " -n1 -s

python3 autotvm.py resnet 0 1000 2>&1 | tee autotvm_resnet_0.log
python3 autotvm.py resnet 1 1000 2>&1 | tee autotvm_resnet_1.log
python3 autotvm.py resnet 2 1000 2>&1 | tee autotvm_resnet_2.log
python3 autotvm.py resnet 3 1000 2>&1 | tee autotvm_resnet_3.log
python3 autotvm.py resnet 4 1000 2>&1 | tee autotvm_resnet_4.log
python3 autotvm.py resnet 5 1000 2>&1 | tee autotvm_resnet_5.log
python3 autotvm.py resnet 6 1000 2>&1 | tee autotvm_resnet_6.log
python3 autotvm.py resnet 7 1000 2>&1 | tee autotvm_resnet_7.log
python3 autotvm.py resnet 8 1000 2>&1 | tee autotvm_resnet_8.log
python3 autotvm.py resnet 9 1000 2>&1 | tee autotvm_resnet_9.log
python3 autotvm.py resnet 10 1000 2>&1 | tee autotvm_resnet_10.log
python3 autotvm.py resnet 11 1000 2>&1 | tee autotvm_resnet_11.log
python3 autotvm.py resnet 12 1000 2>&1 | tee autotvm_resnet_12.log

mkdir -p autotvm_resnet/run_time1_resnet
mv *json *csv *log  autotvm_resnet/run_time1_resnet


# yolo 0-9
python3 autotvm.py yolo 0 1000 2>&1 | tee autotvm_yolo_0.log
python3 autotvm.py yolo 1 1000 2>&1 | tee autotvm_yolo_1.log
python3 autotvm.py yolo 2 1000 2>&1 | tee autotvm_yolo_2.log
python3 autotvm.py yolo 3 1000 2>&1 | tee autotvm_yolo_3.log
python3 autotvm.py yolo 4 1000 2>&1 | tee autotvm_yolo_4.log
python3 autotvm.py yolo 5 1000 2>&1 | tee autotvm_yolo_5.log
python3 autotvm.py yolo 6 1000 2>&1 | tee autotvm_yolo_6.log
python3 autotvm.py yolo 7 1000 2>&1 | tee autotvm_yolo_7.log
python3 autotvm.py yolo 8 1000 2>&1 | tee autotvm_yolo_8.log
python3 autotvm.py yolo 9 1000 2>&1 | tee autotvm_yolo_9.log

mkdir -p autotvm_yolo/run_time1_yolo
mv *json *csv *log  autotvm_yolo/run_time1_yolo

python3 autotvm_mm.py 0 1000 2>&1 | tee mm_0.log
python3 autotvm_mm.py 1 1000 2>&1 | tee mm_1.log
python3 autotvm_mm.py 2 1000 2>&1 | tee mm_2.log
python3 autotvm_mm.py 3 1000 2>&1 | tee mm_3.log
python3 autotvm_mm.py 4 1000 2>&1 | tee mm_4.log
python3 autotvm_mm.py 5 1000 2>&1 | tee mm_5.log

mkdir -p autotvm_mm/run_time1
mv *csv *json *log autotvm_mm/run_time1