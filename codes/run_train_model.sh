python train_video.py -exp vimeo_exp_01_mse_q1024 -n 4 -d /media/sugon/新加卷/wff/vimeo_septuplet --epochs 100 -lr 1e-4 --batch-size 4 --cuda --gpu_id 0 --lambda 1024 --save  #--checkpoint ../experiments/exp_01_mse_q3/checkpoints/checkpoint_best_loss.pth.tar

#python train_video.py -exp exp_01_mse_q5 -m ssf2020 -n 16 -d /yuan/wff/vimeo_septuplet --epochs 600 -lr 1e-4 --batch-size 8 --cuda --gpu_id 7 --lambda 0.032 --save  --checkpoint ../experiments/exp_01_mse_q5/checkpoints/checkpoint_best_loss.pth.tar

#python train_video.py -exp dcvc_exp_01_mse_q04 --lambda 0.04 -d ~/sy/vimeo_septuplet/ -lr 1e-4 --epoch 200 --cuda --gpu_id 0