import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np


class retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l: a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of the first square patch.
    - k: number of patches to extract in the glimpse.
    - s: scaling factor that controls the size of
      successive patches.

    Returns
    -------
    - phi: a 5D tensor of shape (B, k, g, g, C). The
      foveated glimpse of the image.
    """
    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        #phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """
        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        # denormalize coords of patch center
        coords = self.denormalize(H, l)

        # compute top left corner of patch
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)
            T = im.shape[-1]

            # compute slice indices
            from_x = coords[i,0] - (size[i] // 2)
            from_y = coords[i,1] - (size[i] // 2)
 
            to_x = from_x + size[i]
            to_y = from_y + size[i]
            
            # cast to ints
            from_x, to_x = from_x.item(), to_x.item()
            from_y, to_y = from_y.item(), to_y.item()

            # pad tensor in case exceeds
            if self.exceeds(from_x, to_x, from_y, to_y, T):
                pad_dims = (
                    size[i]//2+1, size[i]//2+1,
                    size[i]//2+1, size[i]//2+1,
                    0, 0,
                    0, 0,
                )
                im = F.pad(im, pad_dims, "constant", 0)

                # add correction factor
                from_x += (size[i]//2+1)
                to_x += (size[i]//2+1)
                from_y += (size[i]//2+1)
                to_y += (size[i]//2+1)

            # and finally extract
            patch.append(im[:, :, from_y:to_y, from_x:to_x])

        # concatenate into a single tensor
        # patch = torch.cat(patch)

        # put patches of different size into a dict
        patch_dict = {}
        for num in [32, 16, 8, 4, 2, 1]:
            idx = self.get_index_of_this_size(size, num)
            if len(idx) > 0:
                patch_dict["size_"+str(num)] = torch.cat(
                    [patch[i] for i in idx], 0) 

        return patch_dict

    def get_index_of_this_size(self, size_batch, size):
        # for a batch of size
        # get the idx of a particular size within the batch
        w = (size_batch == size).squeeze().nonzero()
        return w.numpy().flatten().tolist()
        

    def denormalize(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (
            (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T)
        ):
            return True
        return False


def conv3x3(in_planes, out_planes, stride=1, groups=1):
  """
  3x3 convolution with padding
  """
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, groups=groups, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, 
    downsample=None, groups=1, norm_layer=None):
        
    super(BasicBlock, self).__init__()
    
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1:
      raise ValueError('BasicBlock only supports groups=1')
    # Both self.conv1 and self.downsample layers 
    # downsample the input when stride != 1
    
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)

    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out




class glimpse_network(nn.Module):
    """
    A convolution network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:


    Args
    ----
    - block: block in residue layers
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, C, H, W). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, block, h_g, h_l, h_s, g, k, s, c):
        super(glimpse_network, self).__init__()
        self.retina = retina(g, k, s)
        
        # input layers of residue blocks
        self.planes = [3, 12, 24, 48, 96, 192]
        
        # number of basic block per residue layer
        self.num_block = 1
        self.size_options = [2, 4, 8, 16, 32]
        self.size_options_t = torch.Tensor(self.size_options).long()

        # 5 feature extractors for images of different size
        self.feature_extractors = {}
        num_residue_layers = 1
        for input_size in self.size_options:
            res_blocks = []
            for i in range(num_residue_layers):
                res_block = nn.Sequential(
                    self._make_layer(block, self.planes[i], self.num_block),
                    nn.Conv2d(self.planes[i], self.planes[i+1],
                              kernel_size=2, stride=2),
                    nn.BatchNorm2d(self.planes[i+1]),
                    nn.ReLU(inplace=True)
                )
                res_blocks.append(res_block)
            self.feature_extractors['size_'+str(input_size)] = nn.Sequential(
            *res_blocks)
            num_residue_layers = num_residue_layers + 1

        self.feature_encoders = {}
        i = 1
        for input_size in self.size_options:
            self.feature_encoders['size_'+str(input_size)] = nn.Linear(
            self.planes[i], h_g)
            i = i + 1
            
        # encode location
        self.fc_1 = nn.Linear(2, h_l)
        
        # encode size
        self.fc_2 = nn.Linear(5, h_s)

        self.fc_what = nn.Linear(h_g, h_g+h_l+h_s)
        self.fc_where = nn.Linear(h_l, h_g+h_l+h_s)
        self.fc_size = nn.Linear(h_s, h_g+h_l+h_s)


    def forward(self, x, l_t_prev, size_t_prev):
        size_idx = torch.argmax(size_t_prev, 1).view(-1,1)
        size_t_prev_int = self.size_options_t[size_idx]
        phi = self.retina.extract_patch(x, l_t_prev, size_t_prev_int)
        
        feature_batch = []
        for k, v in phi.items():
            f = self.feature_extractors[k](v)
            f = f.view(f.shape[0], -1)
            f = F.relu(self.feature_encoders[k](f))
            feature_batch.append(f)

        # concatenate features into one batch
        feature_batch = torch.cat(feature_batch,0)

        what = self.fc_what(feature_batch)
        
        l_out = F.relu(self.fc_1(l_t_prev))
        where = self.fc_where(l_out)

        s_out = F.relu(self.fc_2(size_t_prev))
        size = self.fc_size(s_out)
        

        # feed to fc layer
        g_t = F.relu(what + where + size)

        return g_t

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` and size 
    of subimage for the next time step

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        # fc layer to decide next location
        self.fc_loc = nn.Linear(input_size, output_size)
        
        # fc layer to decide next size
        # 5 possiblities
        self.fc_size = nn.Linear(input_size, 5)

    def forward(self, h_t):
        # compute mean
        mu = torch.tanh(self.fc_loc(h_t.detach()))

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise

        # bound between [-1, 1]
        l_t = torch.tanh(l_t)

        # size of next patch
        size_t = torch.softmax(self.fc_size(h_t.detach()), dim=1)

        return mu, l_t, size_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t.detach()))
        return b_t
