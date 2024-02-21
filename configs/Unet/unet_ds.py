
"""Training Unet to downsample MRI images."""

from configs.default_configs import get_default_configs



def get_config():
  config = get_default_configs()
  # training
  training = config.training




  # data
  data = config.data
  data.dataset = 'MRI'
  data.train = '/home/arpanp/FusionNet/dataset/train'
  data.eval = '/home/arpanp/FusionNet/dataset/eval'
  data.image_size1 = 720
  data.image_size2 = 512


  # model
  model = config.model
  model.name = 'Unet'
  model.dim=36
  model.in_channels=2
  model.dim_mults=(1,2,4,8)
  model.out_dim=1
  model.ema_rate = 0.999
#   model.normalization = 'GroupNorm'
#   model.nonlinearity = 'swish'
#   model.nf = 128
#   model.ch_mult = (1, 2, 2, 2)
#   model.num_res_blocks = 4
#   model.attn_resolutions = (16,)
#   model.resamp_with_conv = True
#   model.conditional = True
#   model.fir = True
#   model.fir_kernel = [1, 3, 3, 1]
#   model.skip_rescale = True
#   model.resblock_type = 'biggan'
#   model.progressive = 'none'
#   model.progressive_input = 'residual'
#   model.progressive_combine = 'sum'
#   model.attention_type = 'ddpm'
#   model.init_scale = 0.
#   model.fourier_scale = 16
#   model.conv_size = 3
  


  return config
