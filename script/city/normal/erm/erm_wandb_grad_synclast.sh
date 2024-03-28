python main.py --ds city_normal --dt /media/mountHDD3/data_storage/cityscapes/unzip --bs 32 --wk 8 \
    --trdms 0 --tkss seg depth --losses ce mse \
    --m erm --hp ./hparams/erm.json \
    --model hps --at segnet --bb basenano \
    --lr 0.001 --seed 0 \
    --tm sup --dvids 0 --port 7777 \
    --round 1000 --chkfreq 100 --grad \
    --wandb --wandb_prj MTLDOG --wandb_entity heartbeats --synclast