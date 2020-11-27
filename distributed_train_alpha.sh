export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=1
# python -m torch.distributed.launch --nproc_per_node 2 --master_addr="10.20.13.116" --master_port=23456 --node_rank=0 --nnodes=2  train.py --cfg="models/yolov5m.yaml" --weights='weights/v3.1/yolov5m/pt' --data="data/bdd100k.yaml" --hyp="data/bddk.hyp.finetune.yaml" --batch-size=48 --epochs=50

python -m torch.distributed.launch --nproc_per_node 2 --node_rank=0 --nnodes=1 --master_addr="10.20.13.116" --master_port=23456 train.py --cfg="models/yolov5m.yaml" --weights='' --data="data/traffic_bdd100k.yaml" --hyp="experiment/bddk.hyp.scratch_v3.yaml" --batch-size=64 --epochs=300 --sync-bn --name scratch_v3_m --notest --exist-ok
