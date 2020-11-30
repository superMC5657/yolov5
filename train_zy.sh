

python train.py --cfg="experiment/models/yolov5s_v2.yaml" --weights='' --data="data/coco128.yaml" --hyp="experiment/hyp/bddk.hyp.scratch_v3.yaml" --batch-size=12 --epochs=2 --name coco128 --notest --exist-ok