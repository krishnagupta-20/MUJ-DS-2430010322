"""Model definitions: Baseline CNN, CBAM-CNN, ViT-Small, Hybrid Conv-Transformer, ConvNeXt-style.

All five models share the same input (28x28x1), the same in-model augmentation and a
10-way softmax head, so only the backbone differs. `REGISTRY` maps a model name to its
builder, family and the hyper-parameter grid searched in train.py.
"""
import os
from functools import partial

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers, models, ops

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


# ---------------------------------------------------------------- shared pieces
def augmentation():
    """Train-time augmentation (active only when training=True)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
    ], name="augment")


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return layers.BatchNormalization()(x)


# ---------------------------------------------------------------- CBAM (reference CNNs)
class ChannelStats(layers.Layer):
    """Stacks per-pixel mean and max across channels -> (H, W, 2)."""

    def call(self, x):
        return ops.concatenate([ops.mean(x, axis=-1, keepdims=True),
                                ops.max(x, axis=-1, keepdims=True)], axis=-1)

    def compute_output_shape(self, s):
        return (*s[:-1], 2)


def channel_attention(x, ratio=8):
    ch = x.shape[-1]
    d1 = layers.Dense(max(ch // ratio, 4), activation="relu")
    d2 = layers.Dense(ch)
    avg = d2(d1(layers.GlobalAveragePooling2D()(x)))
    mx = d2(d1(layers.GlobalMaxPooling2D()(x)))
    att = layers.Activation("sigmoid")(layers.Add()([avg, mx]))
    att = layers.Reshape((1, 1, ch))(att)
    return layers.Multiply()([x, att])


def spatial_attention(x):
    att = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(ChannelStats()(x))
    return layers.Multiply()([x, att])


def cbam_block(x, ratio=8):
    return spatial_attention(channel_attention(x, ratio))


def build_cnn(attention=None, dropout=0.3, name="CNN"):
    """3 conv blocks (32/64/128), optional CBAM after each, GAP, Dense(128), softmax."""
    inputs = layers.Input(INPUT_SHAPE)
    x = augmentation()(inputs)
    for i, f in enumerate([32, 64, 128]):
        x = conv_block(x, f)
        if attention == "cbam":
            x = cbam_block(x)
        if i < 2:
            x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return models.Model(inputs, layers.Dense(NUM_CLASSES, activation="softmax")(x), name=name)


# ---------------------------------------------------------------- Transformer pieces
class AddPositionEmbedding(layers.Layer):
    """Learned positional embedding added to a (B, N, D) token tensor."""

    def build(self, input_shape):
        self.pos = self.add_weight(shape=(1, input_shape[1], input_shape[2]),
                                   initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                   name="pos_embedding")

    def call(self, x):
        return x + self.pos

    def compute_output_shape(self, s):
        return s


class TransformerBlock(layers.Layer):
    """Pre-norm encoder block: x + MHSA(LN(x)); x + MLP(LN(x))."""

    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.1, **kw):
        super().__init__(**kw)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation="gelu"), layers.Dropout(dropout),
            layers.Dense(dim), layers.Dropout(dropout)])

    def call(self, x, training=None, return_attention=False):
        h = self.norm1(x)
        if return_attention:
            a, scores = self.attn(h, h, return_attention_scores=True, training=training)
        else:
            a = self.attn(h, h, training=training)
        x = x + a
        x = x + self.mlp(self.norm2(x), training=training)
        return (x, scores) if return_attention else x

    def compute_output_shape(self, s):
        return s


def transformer_head(x, depth, dim, heads, mlp_ratio, dropout):
    x = AddPositionEmbedding(name="pos_embed")(x)
    x = layers.Dropout(dropout)(x)
    for i in range(depth):
        x = TransformerBlock(dim, heads, mlp_ratio, dropout, name=f"block_{i}")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(NUM_CLASSES, activation="softmax")(x)


def build_vit(patch_size=4, dim=64, depth=6, heads=4, mlp_ratio=2.0, dropout=0.1, name="ViT"):
    """Pure ViT: patchify (Conv2D stride = patch) -> pos. embedding -> encoder -> mean-pool."""
    inputs = layers.Input(INPUT_SHAPE)
    x = augmentation()(inputs)
    x = layers.Conv2D(dim, patch_size, strides=patch_size, name="patch_embed")(x)
    x = layers.Reshape(((28 // patch_size) ** 2, dim))(x)
    return models.Model(inputs, transformer_head(x, depth, dim, heads, mlp_ratio, dropout), name=name)


def build_hybrid(dim=96, depth=3, heads=4, mlp_ratio=2.0, dropout=0.1, name="Hybrid"):
    """Conv stem (28 -> 7x7 maps = 49 tokens) followed by a shallow transformer encoder."""
    inputs = layers.Input(INPUT_SHAPE)
    x = augmentation()(inputs)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = layers.MaxPooling2D()(x)                       # 14x14
    x = conv_block(x, dim)
    x = layers.MaxPooling2D()(x)                       # 7x7 -> 49 tokens
    x = layers.Reshape((49, dim))(x)
    return models.Model(inputs, transformer_head(x, depth, dim, heads, mlp_ratio, dropout), name=name)


# ---------------------------------------------------------------- ConvNeXt-style
class LayerScale(layers.Layer):
    def __init__(self, init_value=1e-6, **kw):
        super().__init__(**kw)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer=keras.initializers.Constant(self.init_value),
                                     name="gamma")

    def call(self, x):
        return x * self.gamma

    def compute_output_shape(self, s):
        return s


def convnext_block(x, dim, dropout):
    shortcut = x
    x = layers.DepthwiseConv2D(7, padding="same")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(4 * dim, activation="gelu")(x)
    x = layers.Dense(dim)(x)
    x = LayerScale()(x)
    x = layers.Dropout(dropout)(x)
    return layers.Add()([shortcut, x])


def build_convnext(dims=(48, 96), depths=(2, 2), dropout=0.1, name="ConvNeXt"):
    inputs = layers.Input(INPUT_SHAPE)
    x = augmentation()(inputs)
    x = layers.Conv2D(dims[0], 2, strides=2)(x)        # patchify stem -> 14x14
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    for _ in range(depths[0]):
        x = convnext_block(x, dims[0], dropout)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv2D(dims[1], 2, strides=2)(x)        # downsample -> 7x7
    for _ in range(depths[1]):
        x = convnext_block(x, dims[1], dropout)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout)(x)
    return models.Model(inputs, layers.Dense(NUM_CLASSES, activation="softmax")(x), name=name)


# ---------------------------------------------------------------- registry
# model name -> builder, family, hyper-parameter grid searched in train.py
REGISTRY = {
    "Baseline CNN": dict(builder=partial(build_cnn, attention=None), family="CNN",
                         grid={"dropout": [0.1, 0.3]}),
    "CBAM-CNN": dict(builder=partial(build_cnn, attention="cbam"), family="CNN",
                     grid={"dropout": [0.1, 0.3]}),
    "ConvNeXt-style": dict(builder=build_convnext, family="Modern CNN",
                           grid={"dropout": [0.1, 0.3]}),
    "ViT-Small": dict(builder=build_vit, family="Transformer",
                      grid={"dropout": [0.1, 0.3], "patch_size": [4, 7]}),
    "Hybrid Conv-Transformer": dict(builder=build_hybrid, family="CNN+Transformer",
                                    grid={"dropout": [0.1, 0.3]}),
}


def slug(name):
    """File-system friendly model name, e.g. 'ViT-Small' -> 'vit_small'."""
    return name.lower().replace(" ", "_").replace("-", "_")


def build_model(name, **hp):
    return REGISTRY[name]["builder"](name=slug(name), **hp)
