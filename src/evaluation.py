# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapted for 1D protein sequence models.
#

import json
import numpy as np
from logging import getLogger

from .model import update_predictions, flip_attributes, sequence_cross_entropy
from .utils import print_accuracies

logger = getLogger()


class Evaluator(object):
    """
    Evaluation utilities for sequence-based Fader Networks.
    """

    def __init__(self, ae, lat_dis, ptc_dis, clf_dis, eval_clf, data, params):
        """
        Evaluator initialization.
        For the sequence model, eval_clf is unused (kept for interface consistency).
        """
        self.data = data
        self.params = params

        # modules
        self.ae = ae
        self.lat_dis = lat_dis
        self.ptc_dis = ptc_dis
        self.clf_dis = clf_dis
        self.eval_clf = eval_clf  # always None in sequence version

        # sequence-only compatibility check
        assert hasattr(params, "seq_len"), "Evaluator expects params.seq_len for 1D model."
        logger.info(f"Initialized sequence evaluator (seq_len={params.seq_len}).")

    # -------------------------------------------------------
    #  Reconstruction loss
    # -------------------------------------------------------
    def eval_reconstruction_loss(self):
        data = self.data
        params = self.params
        self.ae.eval()
        bs = params.batch_size

        total_loss = 0.0
        total_tokens = 0
        for i in range(0, len(data), bs):
            batch_x, batch_y = data.eval_batch(i, i + bs)
            _, dec_outputs = self.ae(batch_x, batch_y)
            x_hat = dec_outputs[-1]
            # same CE as in training (per-token mean)
            B, V, T = x_hat.shape
            ce = sequence_cross_entropy(x_hat, batch_x)      # per-token mean
            total_loss += ce.item() * (B * T)
            total_tokens += B * T

        return total_loss / max(1, total_tokens)


    # -------------------------------------------------------
    #  Latent discriminator accuracy
    # -------------------------------------------------------
    def eval_lat_dis_accuracy(self):
        """Compute the latent discriminator prediction accuracy."""
        data = self.data
        params = self.params
        self.ae.eval()
        self.lat_dis.eval()
        bs = params.batch_size

        all_preds = [[] for _ in range(len(params.attr))]
        for i in range(0, len(data), bs):
            batch_x, batch_y = data.eval_batch(i, i + bs)
            enc_outputs = self.ae.encode(batch_x)
            preds = self.lat_dis(enc_outputs[-1]).detach().cpu()
            update_predictions(all_preds, preds, batch_y.detach().cpu(), params)

        return [np.mean(x) for x in all_preds]

    # -------------------------------------------------------
    #  Patch discriminator accuracy (optional)
    # -------------------------------------------------------
    def eval_ptc_dis_accuracy(self):
        """Compute the patch discriminator prediction accuracy."""
        data = self.data
        params = self.params
        self.ae.eval()
        self.ptc_dis.eval()
        bs = params.batch_size

        real_preds, fake_preds = [], []
        for i in range(0, len(data), bs):
            batch_x, batch_y = data.eval_batch(i, i + bs)
            flipped = flip_attributes(batch_y, params, "all")
            _, dec_outputs = self.ae(batch_x, flipped)
            real_preds.extend(self.ptc_dis(batch_x).detach().tolist())
            fake_preds.extend(self.ptc_dis(dec_outputs[-1]).detach().tolist())

        return real_preds, fake_preds

    # -------------------------------------------------------
    #  Classifier discriminator accuracy (optional)
    # -------------------------------------------------------
    def eval_clf_dis_accuracy(self):
        """Compute the classifier discriminator prediction accuracy."""
        data = self.data
        params = self.params
        self.ae.eval()
        self.clf_dis.eval()
        bs = params.batch_size

        all_preds = [[] for _ in range(params.n_attr)]
        for i in range(0, len(data), bs):
            batch_x, batch_y = data.eval_batch(i, i + bs)
            enc_outputs = self.ae.encode(batch_x)
            k = 0
            for j, (_, n_cat) in enumerate(params.attr):
                for value in range(n_cat):
                    flipped = flip_attributes(batch_y, params, j, new_value=value)
                    dec_outputs = self.ae.decode(enc_outputs, flipped)
                    preds = self.clf_dis(dec_outputs[-1])[:, j:j + n_cat].max(1)[1].view(-1)
                    all_preds[k].extend((preds.cpu() == value).tolist())
                    k += 1
            assert k == params.n_attr
        return [np.mean(x) for x in all_preds]

    # -------------------------------------------------------
    #  Overall evaluation
    # -------------------------------------------------------
    def evaluate(self, n_epoch):
        """
        Run all evaluations and log metrics.
        """
        params = self.params
        logger.info("")

        # Reconstruction
        ae_loss = self.eval_reconstruction_loss()

        # Latent discriminator
        log_lat_dis = []
        if params.n_lat_dis:
            lat_dis_accu = self.eval_lat_dis_accuracy()
            log_lat_dis.append(("lat_dis_accu", np.mean(lat_dis_accu)))
            for accu, (name, _) in zip(lat_dis_accu, params.attr):
                log_lat_dis.append((f"lat_dis_accu_{name}", accu))
            logger.info("Latent discriminator accuracy:")
            print_accuracies(log_lat_dis)

        # Patch discriminator
        log_ptc_dis = []
        if params.n_ptc_dis:
            ptc_dis_real, ptc_dis_fake = self.eval_ptc_dis_accuracy()
            accu_real = (np.array(ptc_dis_real) >= 0.5).mean()
            accu_fake = (np.array(ptc_dis_fake) <= 0.5).mean()
            log_ptc_dis += [
                ("ptc_dis_preds_real", np.mean(ptc_dis_real)),
                ("ptc_dis_preds_fake", np.mean(ptc_dis_fake)),
                ("ptc_dis_accu_real", accu_real),
                ("ptc_dis_accu_fake", accu_fake),
                ("ptc_dis_accu", (accu_real + accu_fake) / 2),
            ]
            logger.info("Patch discriminator accuracy:")
            print_accuracies(log_ptc_dis)

        # Classifier discriminator
        log_clf_dis = []
        if params.n_clf_dis:
            clf_dis_accu = self.eval_clf_dis_accuracy()
            k = 0
            log_clf_dis += [("clf_dis_accu", np.mean(clf_dis_accu))]
            for name, n_cat in params.attr:
                log_clf_dis.append((f"clf_dis_accu_{name}", np.mean(clf_dis_accu[k:k + n_cat])))
                log_clf_dis.extend(
                    [(f"clf_dis_accu_{name}_{j}", clf_dis_accu[k + j]) for j in range(n_cat)]
                )
                k += n_cat
            logger.info("Classifier discriminator accuracy:")
            print_accuracies(log_clf_dis)

        # Combine logs
        logger.info(f"Autoencoder loss: {ae_loss:.5f}")
        to_log = dict(
            [("n_epoch", n_epoch), ("ae_loss", ae_loss)]
            + log_lat_dis
            + log_ptc_dis
            + log_clf_dis
        )
        logger.debug("__log__:%s" % json.dumps(to_log))
        return to_log


# -----------------------------------------------------------
#  Standalone classifier accuracy computation (optional)
# -----------------------------------------------------------
def compute_accuracy(classifier, data, params):
    """Compute attribute prediction accuracy for a standalone classifier."""
    classifier.eval()
    bs = params.batch_size
    all_preds = [[] for _ in range(len(classifier.attr))]
    for i in range(0, len(data), bs):
        batch_x, batch_y = data.eval_batch(i, i + bs)
        preds = classifier(batch_x).detach().cpu()
        update_predictions(all_preds, preds, batch_y.detach().cpu(), params)
    return [np.mean(x) for x in all_preds]
