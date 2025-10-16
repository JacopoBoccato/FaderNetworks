# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapted by JacopoBoccato for sequence data.

import os
import re
import pickle
import random
import inspect
import argparse
import subprocess
import torch
import torch.optim as optim   # single import; avoids name collisions
from logging import getLogger

from .logger import create_logger
# Do not import AVAILABLE_ATTR from loader; attributes are passed explicitly

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

# Path where experiment checkpoints will be stored
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

logger = getLogger()

def initialize_exp(params):
    """
    Experiment initialization.
    - Creates a dump directory with a unique ID.
    - Saves the parameters to 'params.pkl'.
    - Sets up logging to file and stdout.
    """
    # compute dump directory and save params
    params.dump_path = get_dump_path(params)
    with open(os.path.join(params.dump_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # create a logger and dump params
    logger = create_logger(os.path.join(params.dump_path, 'train.log'))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {v}' for k, v in sorted(vars(params).items())))
    return logger

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    Accepts: 'on'/'off', 'true'/'false', '1'/'0' (case‑insensitive).
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. use 0 or 1")

def attr_flag(s):
    """
    Parse attributes parameters from command line.
    Examples:
        "Smiling,Male"            -> [("Smiling", 2), ("Male", 2)]
        "Age.3,Gender.2"          -> [("Age", 3), ("Gender", 2)]
        "*"                       -> "*" (handled later)
    """
    if s == "*":
        return s
    attr = s.split(',')
    assert len(attr) == len(set(attr)), "duplicate attributes specified"
    attributes = []
    for x in attr:
        if '.' not in x:
            # default: binary attribute (n_cat=2)
            attributes.append((x, 2))
        else:
            name, cats = x.split('.')
            assert name, "attribute name cannot be empty"
            assert cats.isdigit() and int(cats) >= 2, "number of categories must be >= 2"
            attributes.append((name, int(cats)))
    # sort by (n_cat, name) as in original code
    return sorted(attributes, key=lambda x: (x[1], x[0]))

def check_attr(params):
    """
    Validate and set attribute names and number of categories.
    In the sequence version, AVAILABLE_ATTR is not imported; we trust the user‑provided list.
    """
    if params.attr == '*':
        # You can decide how to handle '*' here (e.g. take all labels from dataset).
        # For simplicity, we assume '*' is not used and raise an error.
        raise ValueError("Wildcard '*' for attributes is not supported in the sequence version.")
    else:
        # ensure each attribute has at least 2 categories
        assert all(n_cat >= 2 for _, n_cat in params.attr), "each attribute must have at least 2 categories"
    # compute total number of attribute categories
    params.n_attr = sum(n_cat for _, n_cat in params.attr)

def get_optimizer(model, s):
    """
    Parse optimizer parameters and build the optimizer.
    Accepts strings such as:
        - "sgd,lr=0.01"
        - "adam,lr=0.0002,beta1=0.5,beta2=0.999"
        - "rmsprop,lr=0.001"
    Uses Python 3's inspect.signature() to validate keyword arguments.
    """
    if "," in s:
        method = s[:s.find(',')].lower()
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            key, val = x.split('=')
            key = key.strip()
            val = val.strip()
            assert re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$", val), f"invalid numeric value: {val}"
            optim_params[key] = float(val)
    else:
        method = s.lower()
        optim_params = {}

    # select optimizer
    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        # convert legacy beta1/beta2 to betas tuple
        if 'beta1' in optim_params or 'beta2' in optim_params:
            optim_params['betas'] = (
                optim_params.pop('beta1', 0.5),
                optim_params.pop('beta2', 0.999)
            )
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params, "SGD requires a learning rate (lr=...)"
    else:
        raise Exception(f'Unknown optimization method: "{method}"')

    # validate parameter names
    sig = inspect.signature(optim_fn)
    valid_args = sig.parameters.keys()
    for k in optim_params.keys():
        if k not in valid_args:
            raise Exception(f'Unexpected optimizer parameter "{k}". Expected one of: {list(valid_args)}')

    # create optimizer
    return optim_fn(model.parameters(), **optim_params)

def clip_grad_norm(parameters, max_norm, norm_type=2.0):
    """
    Clip gradient norm of an iterable of parameters in‑place.
    Uses PyTorch 1.x style operations (no .data).
    """
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if not parameters:
        return
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.detach().norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

def get_dump_path(params):
    """
    Create and return a unique directory under MODELS_PATH for storing this experiment.
    """
    assert os.path.isdir(MODELS_PATH), f"Models path does not exist: {MODELS_PATH}"

    # create the experiment subfolder (params.name)
    sweep_path = os.path.join(MODELS_PATH, params.name)
    if not os.path.exists(sweep_path):
        os.makedirs(sweep_path)

    # generate a random 10‑character experiment ID
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        exp_id = ''.join(random.choice(chars) for _ in range(10))
        dump_path = os.path.join(sweep_path, exp_id)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
            break
    return dump_path

def reload_model(model, to_reload, attributes=None):
    """
    Reload a previously trained model from a .pth file.
    Checks that the state_dict keys match and optionally verifies selected attributes.
    """
    assert os.path.isfile(to_reload), f"Checkpoint not found: {to_reload}"
    loaded = torch.load(to_reload)

    # ensure parameter names match
    model_params = set(model.state_dict().keys())
    loaded_params = set(loaded.state_dict().keys())
    if model_params != loaded_params:
        missing = model_params - loaded_params
        extra = loaded_params - model_params
        raise Exception(f"Model parameter mismatch. Missing: {missing}, Extra: {extra}")

    # optionally verify model attributes (e.g. seq_len, n_amino)
    attributes = attributes or []
    for attr in attributes:
        if not hasattr(model, attr):
            raise Exception(f'Attribute "{attr}" not found in current model')
        if not hasattr(loaded, attr):
            raise Exception(f'Attribute "{attr}" not found in loaded model')
        if getattr(model, attr) != getattr(loaded, attr):
            raise Exception(f'Attribute "{attr}" differs between current model ({getattr(model, attr)}) '
                            f'and loaded model ({getattr(loaded, attr)})')

    # copy weights
    for k in model.state_dict().keys():
        if model.state_dict()[k].size() != loaded.state_dict()[k].size():
            raise Exception(f"Tensor size mismatch on {k}: expected {model.state_dict()[k].size()}, "
                            f"got {loaded.state_dict()[k].size()}")
        model.state_dict()[k].copy_(loaded.state_dict()[k])

def print_accuracies(values):
    """
    Pretty print a list of (name, value) pairs as accuracies.
    """
    assert all(len(x) == 2 for x in values)
    for name, value in values:
        logger.info('{:<20}: {:>6}'.format(name, f'{100 * value:.3f}%'))
    logger.info('')

def get_lambda(lmbda, params):
    """
    Compute discriminators' lambda schedule.
    If lambda_schedule = 0, return the base lambda.
    Else ramp linearly from 0 to lambda over lambda_schedule iterations.
    """
    schedule = params.lambda_schedule
    if schedule == 0:
        return lmbda
    else:
        return lmbda * min(params.n_total_iter, schedule) / float(schedule)
