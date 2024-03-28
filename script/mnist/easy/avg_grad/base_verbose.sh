python main.py --ds mnisteasy --dt ./ds/src --bs 64 --wk 8 \
    --trdms 0 1 --tkss rec cls --losses mse ce \
    --m avg_grad --hp ./hparams/erm.json \
    --model hps --at ae --bb base \
    --lr 0.001 --seed 0 \
    --tm sup --dvids 0 --port 8080 \
    --round 5000 --chkfreq 100 --verbose