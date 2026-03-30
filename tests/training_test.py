# test_train_step.py
import torch
from src.model import AutoEncoder, LatentDiscriminator
from src.training import Trainer
from src.loader import DataSampler
from src.utils import check_attr

# --- dummy params ---
class P:
    seq_len=64; n_amino=21; instance_norm=False
    init_fm=32; max_fm=128; n_layers=5; n_skip=1
    deconv_method='convtranspose'; dec_dropout=0.0
    encoder_hidden_dims=[512, 128]
    decoder_hidden_dims=[128, 512]
    dis_hidden_dims=[128, 64]
    hid_dim=128; lat_dis_dropout=0.3
    attr=[("BinaryAttr", 2)]; n_attr=2
    n_lat_dis=1; n_ptc_dis=0; n_clf_dis=0
    lambda_ae=1.0; lambda_lat_dis=1e-4; lambda_ptc_dis=0.0; lambda_clf_dis=0.0
    lambda_schedule=0
    batch_size=4
    ae_optimizer="adam,lr=0.0005"; dis_optimizer="adam,lr=0.0005"
    clip_grad_norm=5.0
    cuda=False
    x_type='onehot'
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

# continuous target test
params_cont = P()
params_cont.label_type = 'continuous'
check_attr(params_cont)
x_cont = torch.rand(N, params_cont.n_amino, params_cont.seq_len)
y_cont = torch.rand(N, params_cont.n_attr)

data_cont = DataSampler(x_cont, y_cont, params_cont)

ae_cont = AutoEncoder(params_cont)
lat_dis_cont = LatentDiscriminator(params_cont)
trainer_cont = Trainer(ae_cont, lat_dis_cont, None, None, data_cont, params_cont)
trainer_cont.lat_dis_step()
trainer_cont.autoencoder_step()
trainer_cont.step(0)

# indexed-X pipeline test
params_idx = P()
params_idx.x_type = 'continuous'
params_idx.label_type = 'continuous'
check_attr(params_idx)
x_idx = torch.rand(N, params_idx.seq_len, dtype=torch.float32)
y_idx = torch.rand(N, params_idx.n_attr, dtype=torch.float32)

data_idx = DataSampler(x_idx, y_idx, params_idx)
ae_idx = AutoEncoder(params_idx)
lat_dis_idx = LatentDiscriminator(params_idx)
trainer_idx = Trainer(ae_idx, lat_dis_idx, None, None, data_idx, params_idx)
trainer_idx.lat_dis_step()
trainer_idx.autoencoder_step()
trainer_idx.step(0)

print("✅ Trainer one-step smoke tests (binary + continuous + continuous X) passed.")
