# resnet BENCH: bash tools/dist_test.sh projects/configs/soap_resnet_bench.py 4
# effi Bench: bash tools/dist_test.sh projects/configs/soap_bench.py efficient_bench.pth 4
# resnet KITTI: bash tools/dist_test.sh projects/configs/soap_resnet_kitti.py resnet_kitti.pth 4
# effi KITTI: bash tools/dist_test.sh projects/configs/soap_kitti.py efficient_kitti.pth 4

CONFIG=$1 
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox \
    # --test-save=results # trun off to evaluation
