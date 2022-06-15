from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        # TODO: q: what rate to go up in channels?
              # q: what rate to go down in dimension?
        n_layers = 3
        dropout = 0.4

        kernel_sizes = [3,3,3]   
        paddings = [1,1,1]
        strides = [2,2,2]

        inner_channels = [in_channels] + [32*8**i for i in range(n_layers - 1)] + [out_channels]
        # the sizes are: 64 -> 32 -> 16 -> 8, this setup shinks by 2 each time
        for i in range(n_layers):
            #TODO: do we need to put batchnorm and dropout in the final layer?
            modules += [nn.Conv2d(inner_channels[i], inner_channels[i+1], kernel_sizes[i], strides[i], paddings[i]),
                        nn.Dropout2d(dropout) ,nn.BatchNorm2d(inner_channels[i+1]), nn.ReLU(),]
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        n_layers = 3
        dropout = 0.4
        kernel_sizes = [3,3,3]
        strides = [2,2,2]
        paddings = [1,1,1]
        # TODO: I don't like having 0 padding on the final image!
        output_paddings = [1,1,1]

        inner_channels = [out_channels] + [32*8**i for i in range(n_layers - 1)] + [in_channels]
        inner_channels.reverse()
        for i in range(n_layers):
            modules += [nn.ConvTranspose2d(inner_channels[i], inner_channels[i+1], kernel_sizes[i], strides[i], paddings[i], output_paddings[i]),
                       nn.ReLU() ,nn.Dropout2d(dropout) ,nn.BatchNorm2d(inner_channels[i+1])]
        # another convolution at the end for the outer zero padding to not matter so much and for the final move not to be
        # relu
        modules += [nn.Conv2d(out_channels, out_channels, 3, padding =1)]
        # ========================
        
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        # H, W = self.features_shape[1:]
        import numpy as np
        self.enc_mean = nn.Linear(np.prod(self.features_shape), z_dim)
        self.enc_log_sigma2 = nn.Linear(np.prod(self.features_shape), z_dim)

        self.fc_decoder = nn.Linear(z_dim, np.prod(self.features_shape))
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x)
        h = h.view(h.shape[0], -1)
        mu = self.enc_mean(h)
        # yah it's log because varience can't be negative. why is it variance and not std though?
        log_sigma2 = self.enc_log_sigma2(h)
        u = torch.randn_like(log_sigma2)

        z = mu + torch.exp(log_sigma2/2) * u

        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    return loss, data_loss, kldiv_loss
