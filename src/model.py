import os, math
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import numpy as np
import keras
from keras import layers, models, ops

def augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
    ], name="augment")

def relative_position_index(w):
    coords = np.stack(np.meshgrid(np.arange(w), np.arange(w), indexing='ij')).reshape(2, -1)
    rel = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0)
    rel[:, :, 0] += w - 1
    rel[:, :, 1] += w - 1
    rel[:, :, 0] *= 2 * w - 1
    return rel.sum(-1).astype('int32')

def shifted_window_mask(res, w, s):
    img = np.zeros((1, res, res, 1), dtype='float32')
    cnt = 0
    for h in (slice(0, -w), slice(-w, -s), slice(-s, None)):
        for ww in (slice(0, -w), slice(-w, -s), slice(-s, None)):
            img[:, h, ww, :] = cnt
            cnt += 1
    win = img.reshape(1, res // w, w, res // w, w, 1).transpose(0, 1, 3, 2, 4, 5).reshape(-1, w * w)
    diff = win[:, None, :] - win[:, :, None]
    return np.where(diff != 0, -100.0, 0.0).astype('float32')

class SwinBlock(layers.Layer):
    def __init__(self, dim, heads, res, window=7, shift=0, mlp_ratio=2.0,
                 dropout=0.0, drop_path=0.0, **kw):
        super().__init__(**kw)
        if res <= window:
            window, shift = res, 0
        assert res % window == 0
        self.dim, self.heads, self.res, self.window, self.shift = dim, heads, res, window, shift
        self.scale = (dim // heads) ** -0.5
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)
        self.attn_drop = layers.Dropout(dropout)
        self.proj_drop = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation='gelu'), layers.Dropout(dropout),
            layers.Dense(dim), layers.Dropout(dropout)])
        self.dp1 = layers.Dropout(drop_path, noise_shape=(None, 1, 1, 1)) if drop_path > 0 else None
        self.dp2 = layers.Dropout(drop_path, noise_shape=(None, 1, 1, 1)) if drop_path > 0 else None
        self.rel_index = relative_position_index(window)
        self.mask = shifted_window_mask(res, window, shift) if shift > 0 else None

    def build(self, input_shape):
        w = self.window
        self.rel_bias_table = self.add_weight(
            shape=((2 * w - 1) ** 2, self.heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02), name='rel_bias_table')
        # Keras 3: sub-layers must be built explicitly when build() is overridden
        N = w * w
        self.norm1.build(input_shape)
        self.qkv.build((None, N, self.dim))
        self.proj.build((None, N, self.dim))
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        self.built = True

    def _partition(self, x):
        w, C = self.window, self.dim
        x = ops.reshape(x, (-1, self.res // w, w, self.res // w, w, C))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return ops.reshape(x, (-1, w * w, C))

    def _reverse(self, x):
        w, C = self.window, self.dim
        x = ops.reshape(x, (-1, self.res // w, self.res // w, w, w, C))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return ops.reshape(x, (-1, self.res, self.res, C))

    def _attention(self, x, training):
        N, H, hd = self.window ** 2, self.heads, self.dim // self.heads
        qkv = ops.reshape(self.qkv(x), (-1, N, 3, H, hd))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.matmul(q * self.scale, ops.transpose(k, (0, 1, 3, 2)))
        bias = ops.take(self.rel_bias_table, self.rel_index.reshape(-1), axis=0)
        bias = ops.transpose(ops.reshape(bias, (N, N, H)), (2, 0, 1))
        attn = attn + ops.expand_dims(bias, 0)
        if self.mask is not None:
            nW = self.mask.shape[0]
            attn = ops.reshape(attn, (-1, nW, H, N, N)) + self.mask[None, :, None, :, :]
            attn = ops.reshape(attn, (-1, H, N, N))
        attn = self.attn_drop(ops.softmax(attn, axis=-1), training=training)
        out = ops.transpose(ops.matmul(attn, v), (0, 2, 1, 3))
        out = ops.reshape(out, (-1, N, self.dim))
        return self.proj_drop(self.proj(out), training=training)

    def call(self, x, training=None):
        h = self.norm1(x)
        if self.shift > 0:
            h = ops.roll(h, (-self.shift, -self.shift), axis=(1, 2))
        h = self._reverse(self._attention(self._partition(h), training))
        if self.shift > 0:
            h = ops.roll(h, (self.shift, self.shift), axis=(1, 2))
        if self.dp1 is not None:
            h = self.dp1(h, training=training)
        x = x + h
        m = self.mlp(self.norm2(x), training=training)
        if self.dp2 is not None:
            m = self.dp2(m, training=training)
        return x + m

    def compute_output_shape(self, s):
        return s

class PatchMerging(layers.Layer):
    def __init__(self, out_dim, **kw):
        super().__init__(**kw)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.reduce = layers.Dense(out_dim, use_bias=False)
        self.out_dim = out_dim
    def build(self, input_shape):
        self.norm.build((*input_shape[:-1], input_shape[-1] * 4))
        self.reduce.build((*input_shape[:-1], input_shape[-1] * 4))
        self.built = True
    def call(self, x):
        x = ops.concatenate([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], axis=-1)
        return self.reduce(self.norm(x))
    def compute_output_shape(self, s):
        return (s[0], s[1] // 2, s[2] // 2, self.out_dim)

def build_swin(patch_size=2, embed_dim=64, depths=(2, 4), heads=(2, 4), window=7,
               mlp_ratio=2.0, dropout=0.0, drop_path=0.1, name='Swin'):
    inputs = layers.Input((28, 28, 1))
    x = augmentation()(inputs)
    x = layers.Conv2D(embed_dim, patch_size, strides=patch_size, name='patch_embed')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res, dim = 28 // patch_size, embed_dim
    total, k = sum(depths), 0
    for si, (d, h) in enumerate(zip(depths, heads)):
        for bi in range(d):
            dpr = drop_path * k / max(total - 1, 1); k += 1
            x = SwinBlock(dim, h, res, window, shift=0 if bi % 2 == 0 else window // 2,
                          mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr,
                          name=f'stage{si}_block{bi}')(x)
        if si < len(depths) - 1:
            x = PatchMerging(dim * 2, name=f'merge{si}')(x)
            res, dim = res // 2, dim * 2
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return models.Model(inputs, layers.Dense(10, activation='softmax')(x), name=name)
