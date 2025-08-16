from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/dingzhaodong/project/BAT/data/got10k_lmdb'
    settings.got10k_path = '/data/dingzhaodong/project/BAT/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/dingzhaodong/project/BAT/data/itb'
    settings.lasot_extension_subset_path_path = '/data/dingzhaodong/project/BAT/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/dingzhaodong/project/BAT/data/lasot_lmdb'
    settings.lasot_path = '/data/dingzhaodong/project/BAT/data/lasot'
    settings.network_path = '/data/dingzhaodong/project/BAT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/dingzhaodong/project/BAT/data/nfs'
    settings.otb_path = '/data/dingzhaodong/project/BAT/data/otb'
    settings.prj_dir = '/data/dingzhaodong/project/BAT'
    settings.result_plot_path = '/data/dingzhaodong/project/BAT/output/test/result_plots'
    settings.results_path = '/data/dingzhaodong/project/BAT/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/dingzhaodong/project/BAT/output'
    settings.segmentation_path = '/data/dingzhaodong/project/BAT/output/test/segmentation_results'
    settings.tc128_path = '/data/dingzhaodong/project/BAT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/dingzhaodong/project/BAT/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/dingzhaodong/project/BAT/data/trackingnet'
    settings.uav_path = '/data/dingzhaodong/project/BAT/data/uav'
    settings.vot18_path = '/data/dingzhaodong/project/BAT/data/vot2018'
    settings.vot22_path = '/data/dingzhaodong/project/BAT/data/vot2022'
    settings.vot_path = '/data/dingzhaodong/project/BAT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

