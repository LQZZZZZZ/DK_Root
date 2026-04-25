import os


class Config(object):
    def __init__(self):
        self.device = "cpu"
        self.input_channels = 33
        self.sequence_length = 40
        self.kernel_size = 3
        self.stride = 1
        self.final_out_channels = 6
        self.num_classes = 6
        self.dropout = 0.3
        self.use_normalization = True

        self.model_type = "Conv"
        self.num_heads = 1
        self.drop_last = True
        self.batch_size = 16

        self.features_len = 8
        self.num_epoch = 150
        self.beta1 = 0.90
        self.beta2 = 0.99
        self.lr = 3e-4

        self.unlabeled_batch_size = 16
        self.unlabeled_num_epoch = 60
        self.unlabeled_lr = 3e-4

        self.aug_method = "normal"
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.Diffusion = Diffusion()
        self.TimeGAN = TimeGAN_Config()

        self.timegan_hidden_dim = 64
        self.timegan_num_layers = 3
        self.visual_path = os.path.join("visual_result")

        self.loss_type = "CE"
        self.GCE_q = 0.3
        self.topn = "1"
        self.acc_thre_level1 = 0.9
        self.acc_thre_level2 = 0.7
        self.acc_thre_level3 = 0.4
        self.unlabel_confidence_threshold = 0.75
        self.whether_overfit_threshold = 0.4
        self.ood_threshold = 0.01


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.1
        self.jitter_scale_ratio_strong = 5
        self.jitter_ratio = 0.8
        self.max_seg = 8
        self.noise_rate = 0.3


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.1
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 3


class Diffusion(object):
    def __init__(self):
        self.lr = 5e-4
        self.num_epochs = 1000
        self.timesteps = 100
        self.diffaug_num_rounds = 10
        self.ddim_steps = 1
        self.weak_high_ratio = 0.125
        self.strong_low_ratio = 0.5
        self.save_path_diffusion_template = os.path.join("checkpoints", "diffusion_seed_{seed}.pth")
        self.save_path_diffusion = os.path.join("checkpoints", "diffusion.pth")
        self.save_path_diffusion_uncond_template = os.path.join("checkpoints", "diffusion_uncond_seed_{seed}.pth")
        self.save_path_diffusion_uncond = None


class TimeGAN_Config(object):
    def __init__(self):
        self.lr = 5e-4
        self.iterations = 10000
        self.weak_temperature = 0.6
        self.strong_temperature = 1.2
        self.save_path_timegan_template = os.path.join("checkpoints", "timegan_seed_{seed}.pth")
        self.save_path_timegan = None
        self.save_path_timegan_uncond_template = os.path.join("checkpoints", "timegan_uncond_seed_{seed}.pth")
        self.save_path_timegan_uncond = None