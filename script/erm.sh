python main.py --ds mnist --dt ./ds/source --bs 64 --wk 12 --pm \
    --trdms 0 1 \
    --tkss rec cls \
    --losses mse ce \
    --m erm --hp ./hparams/erm.json \
    --model hps --at ae --bb base \
    --seed 0 --tm sup \
    --dvids 0 1 \
    --epoch 100 --lr 0.001 \
    --wandb --log --wandb_prj MTLDOG --wandb_entity heartbeats