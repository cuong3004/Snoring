import jax
from jax import numpy as jnp
from jax.nn import softmax
import flax.linen as nn
from flax import linen as nn
from jax import numpy as jnp
from jax.nn import hard_swish
import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.nn import sigmoid
import jax
import jax
from jax import numpy as jnp
from jax.nn import softmax
import flax.linen as nn


class DynamicConv(nn.Module):
    in_channels: int
    out_channels: int
    context_dim: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    padding: int = 0
    groups: int = 1
    att_groups: int = 1
    bias: bool = False
    k: int = 4
    temp_schedule: tuple = (30, 1, 1, 0.05)

    def setup(self):
        assert self.in_channels % self.groups == 0
        
        self.T_max, self.T_min, self.T0_slope, self.T1_slope = self.temp_schedule
        self.temperature = self.T_max

        self.residuals = nn.Dense(features=self.k * self.att_groups)
        self.weight_initializer = nn.initializers.kaiming_normal()
        self.bias_initializer = nn.initializers.zeros
        
        self.weight = self.param('weight', self.weight_initializer,
                                 (self.k, self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        self.weight = self.weight.reshape((1, self.att_groups, self.k, -1))
        
        if self.bias:
            self.bias = self.param('bias', self.bias_initializer, (self.k, self.out_channels))
            self.bias = self.bias.reshape((-1,))

    def __call__(self, x, g=None):
        b, c, f, t = x.shape
        g_c = g.reshape((b, -1))
        residuals = self.residuals(g_c).reshape((b, self.att_groups, 1, -1))
        attention = softmax(residuals / self.temperature, axis=-1)

        aggregate_weight = (attention @ self.weight).transpose((1, 2)).reshape((b, self.out_channels, 
                                                                                 self.in_channels // self.groups, 
                                                                                 self.kernel_size, self.kernel_size))
        aggregate_weight = aggregate_weight.reshape((b * self.out_channels, self.in_channels // self.groups,
                                                     self.kernel_size, self.kernel_size))
        x = x.reshape((1, -1, f, t))
        
        # In Flax, the convolution operations does not include bias handling in the same way as PyTorch
        # so we handle bias separately if required.
        output = jax.lax.conv_general_dilated(x,
                                              aggregate_weight,
                                              window_strides=(self.stride, self.stride),
                                              padding=((self.padding, self.padding), (self.padding, self.padding)),
                                              lhs_dilation=(self.dilation, self.dilation),
                                              rhs_dilation=(1, 1),
                                              dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                                              feature_group_count=self.groups * b)
        if self.bias is not None:
            aggregate_bias = jnp.dot(attention, self.bias).reshape((-1,))
            output += aggregate_bias.reshape((b, self.out_channels, 1, 1))

        output = output.reshape((b, self.out_channels, output.shape[-2], output.shape[-1]))
        return output

    def update_params(self, epoch):
        t0 = self.T_max - self.T0_slope * epoch
        t1 = 1 + self.T1_slope * (self.T_max - 1) / self.T0_slope - self.T1_slope * epoch
        self.temperature = max(t0, t1, self.T_min)
        print(f"Setting temperature for attention over kernels to {self.temperature}")


class DyReLU(nn.Module):
    channels: int
    context_dim: int
    M: int = 2

    @nn.compact
    def __call__(self, x, g):
        raise NotImplementedError("DyReLU should not be called directly, please use DyReLUB instead.")

    def get_relu_coefs(self, g):
        coef_net = nn.Dense(features=2 * self.M)
        theta = coef_net(g)
        theta = 2 * sigmoid(theta) - 1
        return theta

class DyReLUB(DyReLU):

    @nn.compact
    def __call__(self, x, g):
        assert x.shape[1] == self.channels
        assert g is not None
        b, c, f, t = x.shape
        h_c = g.reshape((b, -1))

        theta = self.get_relu_coefs(h_c)

        lambdas = self.param('lambdas', lambda rng, shape: jnp.array([1.] * self.M + [0.5] * self.M).astype(jnp.float32),
                             (2 * self.M,))
        init_v = self.param('init_v', lambda rng, shape: jnp.array([1.] + [0.] * (2 * self.M - 1)).astype(jnp.float32),
                            (2 * self.M,))

        coef_net = nn.Dense(features=2 * self.M * self.channels)
        theta = coef_net(h_c).reshape((-1, self.channels, 1, 1, 2 * self.M))

        relu_coefs = theta * lambdas + init_v
        x_mapped = x[..., jnp.newaxis] * relu_coefs[..., :self.M] + relu_coefs[..., self.M:]

        if self.M == 2:
            result = jnp.maximum(x_mapped[:, :, :, :, 0], x_mapped[:, :, :, :, 1])
        else:
            result = jnp.max(x_mapped, axis=-1)

        return result
    
    



class CoordAtt(nn.Module):
    @nn.compact
    def __call__(self, x, g):
        g_cf, g_ct = g[1], g[2]
        a_f = nn.sigmoid(g_cf)
        a_t = nn.sigmoid(g_ct)
        # Recalibration with channel-frequency and channel-time weights
        out = x * a_f * a_t
        return out

class ContextGen(nn.Module):
    context_dim: int
    in_ch: int
    exp_ch: int
    norm_layer: nn.Module
    stride: int = 1

    @nn.compact
    def __call__(self, x, g):
        # Shared linear layer implemented as a 2D convolution with 1x1 kernel
        joint_conv = nn.Conv(features=self.context_dim, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=False)
        pool_f = pool_t = nn.avg_pool

        # Separate linear layers for Coordinate Attention
        conv_f = nn.Conv(features=self.exp_ch, kernel_size=(1, 1), strides=(1, 1), padding='VALID')
        conv_t = nn.Conv(features=self.exp_ch, kernel_size=(1, 1), strides=(1, 1), padding='VALID')

        cf = jnp.mean(x, axis=3, keepdims=True)  # adaptive avg pool over width
        ct = jnp.mean(x, axis=2, keepdims=True)  # adaptive avg pool over height

        # Apply pooling for stride > 1
        if self.stride > 1:
            pool_f = nn.avg_pool(kernel_size=(3, 1), strides=(self.stride, 1), padding='SAME')
            pool_t = nn.avg_pool(kernel_size=(1, 3), strides=(1, self.stride), padding='SAME')

        g_cat = jnp.concatenate([cf, ct], axis=1)
        # Joint frequency and time sequence transformation
        g_cat = self.norm_layer(joint_conv(g_cat))
        g_cat = hard_swish(g_cat)

        f, t = cf.shape[1], ct.shape[1]
        h_cf, h_ct = jnp.split(g_cat, indices_or_sections=[f], axis=1)
        h_ct = jnp.transpose(h_ct, (0, 2, 1, 3))
        # Pooling over sequence dimension to get context vector
        h_c = jnp.mean(g_cat, axis=1, keepdims=True)
        g_cf = conv_f(pool_f(h_cf))
        g_ct = conv_t(pool_t(h_ct))

        # Return a tuple of the context vector of size H, frequency, and time sequences for Coordinate Attention
        g = (h_c, g_cf, g_ct)
        return g








from flax import linen as nn
from jax import numpy as jnp
from jax.nn import hard_swish

class CoordAtt(nn.Module):
    def setup(self):
        # Typically, setup is used to define submodules, nothing to define for this simple operation.
        pass

    def __call__(self, x, g):
        g_cf, g_ct = g[1], g[2]
        a_f = nn.sigmoid(g_cf)
        a_t = nn.sigmoid(g_ct)
        # Recalibration with channel-frequency and channel-time weights
        out = x * a_f * a_t
        return out

class ContextGen(nn.Module):
    context_dim: int
    in_ch: int
    exp_ch: int
    norm_layer: nn.Module  # You would define this based on the expected norm layer or pass it as a module callable.
    stride: int = 1
    trainning: bool = True

    def setup(self):
        # You need to instantiate your submodules (layers) here in `setup` when not using `nn.compact`
        self.joint_conv = nn.Conv(features=self.context_dim, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=False)
        self.conv_f = nn.Conv(features=self.exp_ch, kernel_size=(1, 1), strides=(1, 1), padding='VALID')
        self.conv_t = nn.Conv(features=self.exp_ch, kernel_size=(1, 1), strides=(1, 1), padding='VALID')
        self.joint_norm = self.norm_layer(not self.trainning)
        # We no longer declare pooling here. Depending on the stride, apply it in the __call__ method.

        if self.stride > 1:
            # sequence pooling for Coordinate Attention
            self.pool_f = lambda x: nn.avg_pool(x, window_shape=(3, 1), strides=(self.stride, 1), padding='SAME')
            self.pool_t = lambda x: nn.avg_pool(x, window_shape=(1, 3), strides=(1, self.stride), padding='SAME')
        else:
            self.pool_f = Identify()
            self.pool_t = Identify()
            
    def __call__(self, x, g):
        cf = jnp.mean(x, axis=2, keepdims=True)
        ct = jnp.mean(x, axis=1, keepdims=True).transpose((0, 2, 1, 3))

        # if self.stride > 1:
        #     cf = nn.avg_pool(cf, kernel_size=(3, 1), strides=(self.stride, 1), padding='SAME')
        #     ct = nn.avg_pool(ct, kernel_size=(1, 3), strides=(1, self.stride), padding='SAME')

        g_cat = jnp.concatenate([cf, ct], axis=2)
        g_cat = self.joint_norm(self.joint_conv(g_cat))
        g_cat = hard_swish(g_cat)

        f, t = cf.shape[2], ct.shape[2]
        h_cf, h_ct = jnp.split(g_cat, indices_or_sections=[f], axis=2)
        h_ct = jnp.transpose(h_ct, (0, 1, 3, 2))
        h_c = jnp.mean(g_cat, axis=2, keepdims=True)

        g_cf = self.conv_f(self.pool_f(h_cf))
        g_ct = self.conv_t(self.pool_t(h_ct))

        return (h_c, g_cf, g_ct)

