import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        in_channels = self.in_size[0]
        n_layers = 4
        out_channels = 1024
        kernel_sizes = [5,5,5,5]   
        paddings = [1,1,1,2]
        strides = [2,2,2,2]

        inner_channels = [in_channels] + [128, 256, 512] + [out_channels]
        bn = [nn.BatchNorm2d(inner_channels[i+1]) for i in range(n_layers)]
        activations = [nn.LeakyReLU(0.2)]*n_layers
        # the sizes are: 64 -> 32 -> 16 -> 8 -> 4, this setup shinks by 2 each time
        for i in range(n_layers):
            #TODO: do we need to put batchnorm and dropout in the final layer?
            
            modules += [nn.Conv2d(inner_channels[i], inner_channels[i+1], kernel_sizes[i], strides[i], paddings[i], bias=True),
                        bn[i], activations[i]]
      
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Linear(out_channels*4*4, 1)

        self.apply(self._init_weights)
        # ========================
    
    def _init_weights(self, module):
        """ from DCGAN paper: Initialize the weights of a module with"""
        if isinstance(module, nn.Conv2d):    
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear):    
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.fc(self.cnn(x).view(x.shape[0], -1))
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        n_layers = 4
        in_channels = 1024

        kernel_sizes = [5,5,5,5]
        strides = [2,2,2,2]
        paddings = [1,2,2,3]
        output_paddings = [0,0,0,1]
        inner_channels = [in_channels] + [512, 256, 128] + [out_channels]

        bn = [nn.BatchNorm2d(inner_channels[i+1]) for i in range(n_layers - 1)] + [nn.Identity()]
        activations =  [nn.ReLU()]*(n_layers - 1) + [nn.Identity()]
        for i in range(n_layers):
            modules += [nn.ConvTranspose2d(inner_channels[i], inner_channels[i+1], kernel_sizes[i], strides[i], paddings[i], output_paddings[i], bias=True),
                       bn[i], activations[i]]
        
        
        self.apply(self._init_weights)
        
        
        # ========================
        
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Linear(z_dim, featuremap_size*featuremap_size*in_channels)
        # ========================

    def _init_weights(self, module):
        """ from DCGAN paper: Initialize the weights of a module with"""
        if isinstance(module, nn.Conv2d):    
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear):    
            module.weight.data.normal_(mean=0.0, std=0.02)
            

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            lat_samples = torch.randn(n, self.z_dim, device=device, requires_grad=with_grad)
            samples = self.forward(lat_samples).to(device)
        
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        cnn_input = self.fc(z)
        cnn_input = cnn_input.view(z.shape[0], -1, 4, 4)
        x = torch.tanh(self.cnn(cnn_input))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels. 
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # so 
    # ====== YOUR CODE: ======
    # this implementation is weird
    data_labels = torch.ones_like(y_data) * data_label + (torch.rand_like(y_data) - 0.5)* label_noise
    generated_labels = torch.ones_like(y_generated) * (1 - data_label) + (torch.rand_like(y_generated) - 0.5)*label_noise
    loss_data = nn.BCEWithLogitsLoss()(y_data, data_labels)
    loss_generated = nn.BCEWithLogitsLoss()(y_generated, generated_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    # why data label? 
    # the CE is: p(data_label)log(delta(psi(z))) + p(gen_label)log(1 - delta(psi(z))) 
    # the data_label class is the one that the descriminator works on. before it was the real now it's the fake, and these classes are
    # REALLY from z which fits the expression of expectation on z
    # הטעויות הכי מפחידות הן אלה שלא שמים לב אליהן. עדיף לחשוב על משהו הרבה זמן מאשר לא לשים לב שהוא שם
    
    generated_labels = data_label*torch.ones_like(y_generated)
    loss = nn.BCEWithLogitsLoss()(y_generated, generated_labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    n = x_data.shape[0]
    dsc_model.zero_grad()

    # use same fake samples for both? why and why not?


    fake_samples = gen_model.sample(n, with_grad=False)
    dsc_real_scores = dsc_model(x_data)
    dsc_fake_scores = dsc_model(fake_samples)
    print(f'\n{dsc_real_scores.item()=}, {dsc_fake_scores.item()=}')
    dsc_loss = dsc_loss_fn(dsc_real_scores, dsc_fake_scores)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_model.zero_grad()
    
    fake_samples = gen_model.sample(n, with_grad=True)
    dsc_fake_scores = dsc_model(fake_samples)
    gen_loss = gen_loss_fn(dsc_fake_scores)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======

    raise NotImplementedError()
    # ========================

    return saved
