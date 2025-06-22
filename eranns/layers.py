

import tensorflow.keras as keras

class ARB(keras.layers.Layer):
    def __init__(self, stride_freq, stride_time, channels):
        super(ARB, self).__init__()
        self.stride_freq = stride_freq
        self.stride_time = stride_time
        self.channels = channels

        self.k1_freq, self.k2_freq, self.padding_freq = self.get_kernels_and_padding(stride_freq)
        self.k1_time, self.k2_time, self.padding_time = self.get_kernels_and_padding(stride_time)

        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.LeakyReLU(alpha=0.01)

        self.pad1 = keras.layers.ZeroPadding2D(padding=(1, 1))
        self.conv1 = keras.layers.Conv2D(
            channels,
            kernel_size=(self.k1_freq, self.k1_time),
            strides=(stride_freq, stride_time),
            padding="valid",
            use_bias=False
        )

        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.LeakyReLU(alpha=0.01)

        self.pad2 = keras.layers.ZeroPadding2D(padding=(self.padding_freq, self.padding_time))
        self.conv2 = keras.layers.Conv2D(
            channels,
            kernel_size=(self.k2_freq, self.k2_time),
            strides=(1, 1),
            padding="valid",
            use_bias=False
        )

        self.shortcut_conv = None  # definida dinamicamente em build()

    def get_kernels_and_padding(self, stride):
        if stride in [1, 2]:
            return 3, 3, 1
        elif stride == 4:
            return 6, 5, 2
        else:
            raise ValueError(f"Stride {stride} não suportado. Use 1, 2 ou 4.")

    def build(self, input_shape):
        input_channels = input_shape[-1]

        if (self.stride_freq != 1 or self.stride_time != 1) or (input_channels != self.channels):
            self.shortcut_conv = keras.layers.Conv2D(
                self.channels,
                kernel_size=1,
                strides=(self.stride_freq, self.stride_time),
                padding="valid",
                use_bias=False
            )
        else:
            self.shortcut_conv = lambda x: x  # identidade

    def call(self, x):
        shortcut = self.shortcut_conv(x)

        x = self.bn1(x)
        x = self.act1(x)
        x = self.pad1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.pad2(x)
        x = self.conv2(x)

        return keras.layers.Add()([x, shortcut])