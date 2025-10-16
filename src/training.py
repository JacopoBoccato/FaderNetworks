# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapted by JacopoBoccato for protein sequences instead of images.

import os
import numpy as np
import torch
from torch.nn import functional as F
from logging import getLogger

from .utils import get_optimizer, clip_grad_norm, get_lambda, reload_model
from .model import get_attr_loss, flip_attributes

logger = getLogger()


class Trainer(object):
    """
    Trainer for the sequence‑based Fader autoencoder.
    Handles the autoencoder, latent discriminator, optional patch/classifier discriminators.
    Uses DataSampler.next_batch() (not train_batch) to fetch mini‑batches.
    """

    def __init__(self, ae, lat_dis, ptc_dis, clf_dis, data, params):
        # dataset / hyper‑params
        self.data = data
        self.params = params

        # modules
        self.ae = ae
        self.lat_dis = lat_dis
        self.ptc_dis = ptc_dis
        self.clf_dis = clf_dis

        # optimizers
        self.ae_optimizer = get_optimizer(ae, params.ae_optimizer)
        logger.info(ae)
        logger.info('%i parameters in the autoencoder.' %
                    sum(p.nelement() for p in ae.parameters()))
        if params.n_lat_dis:
            logger.info(lat_dis)
            logger.info('%i parameters in the latent discriminator.' %
                        sum(p.nelement() for p in lat_dis.parameters()))
            self.lat_dis_optimizer = get_optimizer(lat_dis, params.dis_optimizer)
        if params.n_ptc_dis:
            logger.info(ptc_dis)
            logger.info('%i parameters in the patch discriminator.' %
                        sum(p.nelement() for p in ptc_dis.parameters()))
            self.ptc_dis_optimizer = get_optimizer(ptc_dis, params.dis_optimizer)
        if params.n_clf_dis:
            logger.info(clf_dis)
            logger.info('%i parameters in the classifier discriminator.' %
                        sum(p.nelement() for p in clf_dis.parameters()))
            self.clf_dis_optimizer = get_optimizer(clf_dis, params.dis_optimizer)

        # reload pretrained models (use sequence‑specific keys)
        if params.ae_reload:
            reload_model(ae, params.ae_reload,
                         ['seq_len', 'n_amino', 'init_fm', 'n_layers',
                          'n_skip', 'attr', 'n_attr'])
        if params.lat_dis_reload:
            reload_model(lat_dis, params.lat_dis_reload,
                         ['seq_len', 'n_amino', 'init_fm', 'max_fm',
                          'n_layers', 'n_skip', 'hid_dim', 'n_attr'])
        if params.ptc_dis_reload:
            reload_model(ptc_dis, params.ptc_dis_reload,
                         ['seq_len', 'n_amino', 'init_fm', 'max_fm'])
        if params.clf_dis_reload:
            reload_model(clf_dis, params.clf_dis_reload,
                         ['seq_len', 'n_amino', 'init_fm', 'max_fm',
                          'hid_dim', 'attr', 'n_attr'])

        # statistics
        self.stats = {
            'rec_costs': [],
            'lat_dis_costs': [],
            'ptc_dis_costs': [],
            'clf_dis_costs': []
        }
        # best metrics
        self.best_loss = 1e12
        self.best_accu = -1e12
        self.params.n_total_iter = 0

    # ======= Latent discriminator training =======
    def lat_dis_step(self):
        """Train latent discriminator for one batch."""
        data = self.data
        params = self.params
        self.ae.eval()
        if not params.n_lat_dis:
            return
        self.lat_dis.train()

        # fetch batch
        batch_x, batch_y = data.next_batch()

        # encode with no grad through AE (frozen)
        with torch.no_grad():
            enc_outputs = self.ae.encode(batch_x)
            z = enc_outputs[-1 - params.n_skip]   # latent features

        preds = self.lat_dis(z)

        # compute discriminator loss on true labels
        loss = get_attr_loss(preds, batch_y, flip=False, params=params)
        self.stats['lat_dis_costs'].append(loss.item())

        # optimize
        self.lat_dis_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.lat_dis.parameters(), params.clip_grad_norm)
        self.lat_dis_optimizer.step()

    # ======= Patch discriminator training =======
    def ptc_dis_step(self):
        """Train patch discriminator (1‑D) for one batch."""
        data = self.data
        params = self.params
        if not params.n_ptc_dis:
            return
        self.ae.eval()
        self.ptc_dis.train()

        batch_x, batch_y = data.next_batch()
        # flip attributes and decode
        flipped = flip_attributes(batch_y, params, 'all')
        with torch.no_grad():
            _, dec_outputs = self.ae(batch_x, flipped)

        real_preds = self.ptc_dis(batch_x)
        fake_preds = self.ptc_dis(dec_outputs[-1])

        y_fake = torch.full_like(real_preds, params.smooth_label)
        if params.cuda:
            y_fake = y_fake.cuda()

        # binary cross‑entropy (real vs fake)
        loss = F.binary_cross_entropy(real_preds, 1 - y_fake)
        loss += F.binary_cross_entropy(fake_preds, y_fake)
        self.stats['ptc_dis_costs'].append(loss.item())

        self.ptc_dis_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.ptc_dis.parameters(), params.clip_grad_norm)
        self.ptc_dis_optimizer.step()

    # ======= Classifier discriminator training =======
    def clf_dis_step(self):
        """Train classifier discriminator for one batch."""
        data = self.data
        params = self.params
        if not params.n_clf_dis:
            return
        self.clf_dis.train()

        batch_x, batch_y = data.next_batch()
        preds = self.clf_dis(batch_x)

        loss = get_attr_loss(preds, batch_y, flip=False, params=params)
        self.stats['clf_dis_costs'].append(loss.item())

        self.clf_dis_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.clf_dis.parameters(), params.clip_grad_norm)
        self.clf_dis_optimizer.step()

    # ======= Autoencoder + adversarial step =======
    def autoencoder_step(self):
        """
        Train the autoencoder with reconstruction + adversarial terms.
        - Reconstruction: MSE between input and output one‑hot sequences.
        - Latent adversarial: fool latent discriminator.
        - Optional patch/classifier discriminator feedback on decoded sequences with flipped labels.
        """
        data = self.data
        params = self.params
        self.ae.train()
        if params.n_lat_dis:
            self.lat_dis.eval()
        if params.n_ptc_dis:
            self.ptc_dis.eval()
        if params.n_clf_dis:
            self.clf_dis.eval()

        batch_x, batch_y = data.next_batch()

        # forward pass through autoencoder
        enc_outputs, dec_outputs = self.ae(batch_x, batch_y)
        recon = dec_outputs[-1]

        # reconstruction loss (MSE on one‑hot)
        rec_loss = ((batch_x - recon) ** 2).mean()
        loss = params.lambda_ae * rec_loss
        self.stats['rec_costs'].append(rec_loss.item())

        # latent adversarial loss (encoder tries to fool discriminator)
        if params.lambda_lat_dis:
            lat_dis_preds = self.lat_dis(enc_outputs[-1 - params.n_skip])
            lat_dis_loss = get_attr_loss(lat_dis_preds, batch_y, flip=True, params=params)
            loss = loss + get_lambda(params.lambda_lat_dis, params) * lat_dis_loss

        # decode with random attributes for patch/classifier discriminators
        if params.lambda_ptc_dis + params.lambda_clf_dis > 0:
            flipped = flip_attributes(batch_y, params, 'all')
            dec_outputs_flipped = self.ae.decode(enc_outputs, flipped)

        # patch discriminator loss for AE (if enabled)
        if params.lambda_ptc_dis:
            ptc_dis_preds = self.ptc_dis(dec_outputs_flipped[-1])
            y_fake = torch.full_like(ptc_dis_preds, params.smooth_label)
            if params.cuda:
                y_fake = y_fake.cuda()
            ptc_dis_loss = F.binary_cross_entropy(ptc_dis_preds, 1 - y_fake)
            loss = loss + get_lambda(params.lambda_ptc_dis, params) * ptc_dis_loss

        # classifier discriminator loss for AE (if enabled)
        if params.lambda_clf_dis:
            clf_dis_preds = self.clf_dis(dec_outputs_flipped[-1])
            clf_dis_loss = get_attr_loss(clf_dis_preds, flipped, flip=False, params=params)
            loss = loss + get_lambda(params.lambda_clf_dis, params) * clf_dis_loss

        # safety check: detect NaNs
        if torch.isnan(loss):
            logger.error("NaN detected in loss; aborting.")
            exit()

        # optimize AE
        self.ae_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.ae.parameters(), params.clip_grad_norm)
        self.ae_optimizer.step()

    # ======= Logging step =======
    def step(self, n_iter):
        """Aggregate and print running statistics every 25 iterations."""
        if len(self.stats['rec_costs']) >= 25:
            mean_loss = [
                ('Latent discriminator', 'lat_dis_costs'),
                ('Patch discriminator', 'ptc_dis_costs'),
                ('Classifier discriminator', 'clf_dis_costs'),
                ('Reconstruction loss', 'rec_costs'),
            ]
            logger.info(('%06i - ' % n_iter) +
                        ' / '.join('%s : %.5f' %
                                    (name, np.mean(self.stats[key]))
                                    for name, key in mean_loss
                                    if self.stats[key]))
            # reset running stats
            for key in self.stats:
                self.stats[key].clear()

        self.params.n_total_iter += 1

    # ======= Model saving =======
    def save_model(self, name):
        """Save the autoencoder and discriminators."""
        def save(module, tag):
            path = os.path.join(self.params.dump_path,
                                f'{name}_{tag}.pth')
            logger.info('Saving %s to %s ...' % (tag, path))
            torch.save(module, path)

        save(self.ae, 'ae')
        if self.params.n_lat_dis:
            save(self.lat_dis, 'lat_dis')
        if self.params.n_ptc_dis:
            save(self.ptc_dis, 'ptc_dis')
        if self.params.n_clf_dis:
            save(self.clf_dis, 'clf_dis')

    def save_best_periodic(self, to_log):
        """Save the best models and periodic checkpoints."""
        if to_log['ae_loss'] < self.best_loss:
            self.best_loss = to_log['ae_loss']
            logger.info('Best reconstruction loss: %.5f' % self.best_loss)
            self.save_model('best_rec')
        if self.params.eval_clf and np.mean(to_log['clf_accu']) > self.best_accu:
            self.best_accu = np.mean(to_log['clf_accu'])
            logger.info('Best evaluation accuracy: %.5f' % self.best_accu)
            self.save_model('best_accu')
        if to_log['n_epoch'] % 5 == 0 and to_log['n_epoch'] > 0:
            self.save_model(f'periodic-{to_log["n_epoch"]}')


# ======= Standalone classifier training =======
def classifier_step(classifier, optimizer, data, params, costs):
    """Train an external classifier on sequence data."""
    classifier.train()
    bs = params.batch_size
    batch_x, batch_y = data.next_batch()
    preds = classifier(batch_x)
    loss = get_attr_loss(preds, batch_y, flip=False, params=params)
    costs.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    if params.clip_grad_norm:
        clip_grad_norm(classifier.parameters(), params.clip_grad_norm)
    optimizer.step()
