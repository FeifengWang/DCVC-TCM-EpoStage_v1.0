#python eval_model_video.py checkpoint -a ssf2020 -exp exp_01_mse_q3 --dataset /bao/wff/media/HEVC_D --output ../result/exp_01_mse_q3 --cuda --gpu_id 1
python eval_model_video.py pretrained -a ssf2020 -q 1 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1 --frames 100
python eval_model_video.py pretrained -a ssf2020 -q 2 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1 --frames 100
python eval_model_video.py pretrained -a ssf2020 -q 3 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1 --frames 100
python eval_model_video.py pretrained -a ssf2020 -q 4 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1 --frames 100
python eval_model_video.py pretrained -a ssf2020 -q 5 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1 --frames 100
#python eval_model_video.py pretrained -a ssf2020 -q 6 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1
#python eval_model_video.py pretrained -a ssf2020 -q 7 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1
#python eval_model_video.py pretrained -a ssf2020 -q 8 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1
#python eval_model_video.py pretrained -a ssf2020 -q 9 --dataset /bao/wff/media/HEVC_D --output ../result/pretrained --cuda --gpu_id 1
