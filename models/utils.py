
"""All functions and modules related to model definition.
"""
import torch
from models.Unet import Unet


def get_model(model_name,config):
    """Get the model class based on `model_name`."""
    if model_name == 'Unet':
        return Unet(dim=config.model.dim,in_channels=config.model.in_channels,dim_mults=config.model.dim_mults, out_dim=config.model.out_dim) 
    else:
        raise NotImplementedError(f'Model {model_name} not implemented yet!')
    
def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  model = get_model(model_name,config)
  model = model.to(config.device)
  return model

def get_model_fn(model, train=False):
  """Create a function to give the output of the model.
  Args:
    model: The  model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """

  def model_fn(x1,x2):
    """Compute the output of the model.

    Args:
      x: A mini-batch of input data.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x1,x2)
    else:
      model.train()
      return model(x1,x2)

  return model_fn