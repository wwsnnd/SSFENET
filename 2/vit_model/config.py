from vit_model.vit_seg_configs import CONFIGS as VIT_CONFIGS

# 添加：允许通过属性方式访问 dict
class AttrDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value

class Config:
    data_dir = "/home/bitmhsi/danguancancer"
    hyperspectral_dir = f"{data_dir}/image_256"
    rgb_dir = f"{data_dir}/converted_rgb"
    label_dir = f"{data_dir}/label_1"
    train_list = f"{data_dir}/train-new.txt"
    test_list = f"{data_dir}/test-new.txt"

    num_channels_hsi = 60
    num_channels_rgb = 3
    image_size = 256
    num_classes = 2
    batch_size = 8
    num_epochs = 500
    learning_rate = 1e-4
    warmup_ratio = 0.05
    initial_lr = 1e-5
    num_workers = 32
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model_name = "VisionTransformer"

    # 配置模型结构
    model_config = AttrDict(VIT_CONFIGS['R50-ViT-B_16'])
    model_config.hidden_size = 256
    model_config.in_channels = num_channels_rgb        # ✅ 明确添加
    model_config.in_channels_hsi = num_channels_hsi    # ✅ 可选添加（若 hybrid 模式用到）
    model_config.transformer["num_heads"] = 8
    model_config.transformer["num_layers"] = 6
    model_config.transformer["mlp_dim"] = 1024
    model_config.transformer["attention_dropout_rate"] = 0.0
    model_config.transformer["dropout_rate"] = 0.1
    model_config.patches["size"] = (16, 16)
    model_config.patches["grid"] = (16, 16)
    model_config.resnet = {
        "num_layers": [3, 4, 6, 3],
        "width_factor": 1.0
    }
    model_config.decoder_channels = [32, 16, 8]
    model_config.n_classes = num_classes
    model_config.n_skip = 3
    model_config.skip_channels = [256, 256, 128]
    model_config.use_ssfenet = True
    model_config.classifier = 'seg'
    model_config.activation = 'softmax'
    model_config.upsampling = 32

    class_weights = [0.1, 3.5]
    pretrained_path = None
    checkpoint_dir = "/home/bitmhsi/myarc1/checkpoints"
    log_dir = "./logs"
    accumulation_steps = 4
    validation_freq = 50
