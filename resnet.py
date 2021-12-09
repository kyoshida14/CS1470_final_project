
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data

# tf.keras.applications.resnet50.ResNet50(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000, **kwargs
# )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=1, bias=False)
    return tf.keras.layers.Conv2d(out_planes, strides=(stride, stride),
                    kernel_size = 3,use_bias=False) #


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return tf.keras.layers.Conv2d(out_planes, strides=(stride, stride),
                    kernel_size = 1,use_bias=False) #


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(planes, planes, stride) #conv3x3(inplanes, planes, stride)
        self.bn1 = tf.keras.layers.BatchNormalization(planes, momentum=0.1, epsilon=0.1e-05)#nn.BatchNorm2d(planes)
        self.relu = tf.keras.layers.ReLU()#nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = tf.keras.layers.BatchNormalization(planes, momentum=0.1, epsilon=0.1e-05)#nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
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


class Bottleneck(tf.keras.Model):
	expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = tf.keras.layers.BatchNormalization(planes, momentum=0.1, epsilon=0.1e-05) #nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = tf.keras.layers.BatchNormalization(planes, momentum=0.1, epsilon=0.1e-05)
        self.conv3 = conv1x1(planes * self.expansion)
        self.bn3 = tf.keras.layers.BatchNormalization(planes * self.expansion, momentum=0.1, epsilon=0.1e-05)
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride
    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(tf.keras.Model):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.padding1 = keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,padding="same"use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal())
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.1e-05) #nn.BatchNorm2d(64)
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D(3, strides=2, padding="valid")  #padding=1
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))
        self.fc = tf.keras.layers.Dense(num_classes)

        # for m in self.modules():
        #     if isinstance(m, tf.keras.layers.Conv2D):
        #         self.initializer = tf.keras.initializers.HeNormal()
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = tf.keras.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return tf.keras.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    if pretrained:
        model = tf.keras.models.Model(inputs=model.input, outputs=predictions)
    return model

    def resnet50(in_channels=3, num_classes=1000):
        return ResNet(ResBottleneckBlock, [3, 4, 6, 3], in_channels, num_classes=num_classes)
