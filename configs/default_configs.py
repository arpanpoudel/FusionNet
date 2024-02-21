import ml_collections
import torch

def get_default_configs():
    config = ml_collections.ConfigDict()
    
    #training configuration
    
    config.training = training=ml_collections.ConfigDict()
    training.batch_size = 2
    training.epochs = 100
    training.loss = 'perceptual'
    training.combined_loss = False
    training.log_freq = 25
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 500
    training.save_every=10
    training.snapshot = True
    
    

    
    #data
    config.data = data= ml_collections.ConfigDict()
    data.dataset = 'MRI'
    data.num_channels = 1
    
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.
    
    #model
    # model
    config.model = model = ml_collections.ConfigDict()
    
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    
    return config

    
    