class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/dingzhaodong/project/BAT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/dingzhaodong/project/BAT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/dingzhaodong/project/BAT/pretrained_networks'
        self.got10k_val_dir = '/data/dingzhaodong/project/BAT/data/got10k/val'
        self.lasot_lmdb_dir = '/data/dingzhaodong/project/BAT/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/dingzhaodong/project/BAT/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/dingzhaodong/project/BAT/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/dingzhaodong/project/BAT/data/coco_lmdb'
        self.coco_dir = '/data/dingzhaodong/project/BAT/data/coco'
        self.lasot_dir = '/data/dingzhaodong/project/BAT/data/lasot'
        self.got10k_dir = '/data/dingzhaodong/project/BAT/data/got10k/train'
        self.trackingnet_dir = '/data/dingzhaodong/project/BAT/data/trackingnet'
        self.depthtrack_dir = '/data/dingzhaodong/project/BAT/data/depthtrack/train'
        self.lasher_dir = '/data/dingzhaodong/project/BAT/data/LasHeR'
        self.visevent_dir = '/data/dingzhaodong/project/BAT/data/visevent/train'
