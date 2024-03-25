python main.py --ds mnist --dt ./ds/src --bs 64 --wk 8 \
    --trdms 0 1 \
    --tkss rec cls \
    --losses mse ce \
    --m erm --hp ./hparams/erm.json \
    --model hps --at ae --bb debug \
    --seed 0 --tm sup \
    --dvids 0 \
    --round 2 --chkfreq 1 --lr 0.001 \
    # --wandb --wandb_prj MTLDOG --wandb_entity heartbeats