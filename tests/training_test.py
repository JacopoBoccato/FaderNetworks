# test_train_step.py
import torch
from src.model import AutoEncoder, LatentDiscriminator
from src.training import Trainer
from src.loader import DataSampler

# --- dummy params ---
class P:
    seq_len=64; n_amino=21; instance_norm=False
    init_fm=32; max_fm=128; n_layers=5; n_skip=1
    deconv_method='convtranspose'; dec_dropout=0.0
    hid_dim=128; lat_dis_dropout=0.3
    attr=[("BinaryAttr", 2)]; n_attr=2
    n_lat_dis=1; n_ptc_dis=0; n_clf_dis=0
    lambda_ae=1.0; lambda_lat_dis=1e-4; lambda_ptc_dis=0.0; lambda_clf_dis=0.0
    lambda_schedule=0
    batch_size=4
    ae_optimizer="adam,lr=0.0005"; dis_optimizer="adam,lr=0.0005"
    clip_grad_norm=5.0
    cuda=False
    dump_path="."
    n_total_iter=0

    # ✅ Add these dummy attributes
    ae_reload = ""
    lat_dis_reload = ""
    ptc_dis_reload = ""
    clf_dis_reload = ""
    eval_clf = ""

params = P()

# --- toy data (8 sequences, 1 binary attr) ---
N = 8
x = torch.rand(N, params.n_amino, params.seq_len)
y = torch.randint(0, 2, (N, params.n_attr)).float()
data = DataSampler(x, y, params)

# --- models ---
ae = AutoEncoder(params)
lat_dis = LatentDiscriminator(params)

trainer = Trainer(ae, lat_dis, None, None, data, params)

# one latent dis step + one AE step
trainer.lat_dis_step()
trainer.autoencoder_step()
trainer.step(0)

print("✅ Trainer one-step smoke test passed.")
