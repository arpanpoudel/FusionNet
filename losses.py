"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
import numpy as np
import time
from models import utils as mutils
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision import models as m 
from torch import nn

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


#Neural Style Transfer
# style_layers, content_layers = [0, 5, 10, 19, 28], [25]
# pretrained_net = m.vgg19(pretrained=True)
# net = nn.Sequential(*[pretrained_net.features[i] for i in
#                       range(max(content_layers + style_layers) + 1)])

# def extract_features(X, content_layers, style_layers):
#     contents = []
#     styles = []
#     for i in range(len(net)):
#         X = net[i](X)
#         if i in style_layers:
#             styles.append(X)
#         if i in content_layers:
#             contents.append(X)
#     return contents, styles
  
# def get_contents(image):
#     return extract_features(image, content_layers, style_layers)[0]

# def get_styles(image):
#     return extract_features(image, content_layers, style_layers)[1]

# def content_loss(Y_hat, Y):
#     return F.mse_loss(Y_hat, Y)

# def gram(X):
#     num_channels, n = X.shape[1], X.numel() // X.shape[1]
#     X = X.view(num_channels, n)
#     return torch.matmul(X, X.t()) / (num_channels * n)

# def style_loss(Y_hat, gram_Y):
#     return F.mse_loss(gram(Y_hat), gram_Y)

# def tv_loss(Y_hat):
#     return 0.5 * (F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :]) +
#                  F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1]))

# def compute_loss(X, contents_Y, styles_Y, contents_Y_hat, styles_Y_hat,configs):
#     # Calculate the content, style, and total variation losses
#     contents_l = [content_loss(Y_hat, Y) for Y_hat, Y in zip(
#         contents_Y_hat, contents_Y)]
#     styles_l = [style_loss(Y_hat, Y) for Y_hat, Y in zip(
#         styles_Y_hat, styles_Y)]
#     tv_l = tv_loss(X)
#     # Sum all of the losses
#     l = (torch.mean(torch.tensor(configs.training.alpha) * torch.tensor(contents_l)) +
#          torch.mean(torch.tensor(configs.training.beta) * torch.tensor(styles_l)) + 
#          torch.tensor(configs.training.gamma) * tv_l)
#     return contents_l, styles_l, tv_l, l


#VGG feature extraction
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = m.vgg16(pretrained=True)
        self.features = torch.nn.Sequential(*list(vgg.features)[:23])  

    def forward(self, x):
        return self.features(x)


# Define the Perceptual Loss
def perceptual_loss(pred, target, feature_extractor):
    if pred.size(1) == 1:  # Check if the input is grayscale (1 channel)
        pred = pred.repeat(1, 3, 1, 1)  # Repeat the channel 3 times
    if target.size(1) == 1:  # Check if the target is grayscale (1 channel)
        target = target.repeat(1, 3, 1, 1)  # Repeat the channel 3 times
        
        
    pred_features = feature_extractor(pred)
    target_features = feature_extractor(target)
    return F.mse_loss(pred_features, target_features)

#ssim loss
def ssim_loss(pred, target, max_val=1.0):
    return 1 - ssim(pred, target, data_range=max_val)

def combined_loss(pred, target, alpha=0.5):
  mse = F.mse_loss(pred, target)
  ssim_loss_val = ssim_loss(pred, target)
  return alpha * mse + (1 - alpha) * ssim_loss_val

def combined_perceptual_loss(pred, target, alpha=0.5, feature_extractor=None):
  mse = F.mse_loss(pred, target)
  perceptual = perceptual_loss(pred, target, feature_extractor)
  return alpha * mse + (1 - alpha) * perceptual

def get_loss_fn(model, train,configs):
  """Create a loss function for training with arbirary model.

  Args:
    model: An `model` object that represents the forward pass.
    train: `True` for training loss and `False` for evaluation loss.
  Returns:
    A loss function.
  """
  loss_function = {
    'mse': F.mse_loss,
    'ssim': ssim_loss,
    'combined': combined_loss,
    'perceptual': combined_perceptual_loss
  }
  #vgg feature extractor
  feature_extractor = VGGFeatureExtractor().eval().to(configs.device)
  

  def loss_fn(model,batch):
    """Compute the loss function.
    Args:
      model: A model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    model_fn=mutils.get_model_fn(model, train)
    output = model_fn(batch['image_1'],batch['image_2'])
    
    
    # Compute the loss using the specified loss function from the configs
    loss_fn = loss_function[configs.training.loss]
    
    if configs.training.loss == 'perceptual':
      loss = loss_fn(output, batch['label'], feature_extractor=feature_extractor,alpha=0.95)
    else:
      loss=loss_fn(output, batch['label'])
      
    return loss
  return loss_fn


def get_step_fn(model, train,configs, optimize_fn=None):
  """Create a one-step training/evaluation function.

  Args:
    model: An model object that represents the forward SDE.
    optimize_fn: An optimization function.

  Returns:
    A one-step function for training or evaluation.
  """
  loss_fn = get_loss_fn(model, train,configs)

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model,batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model,batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
