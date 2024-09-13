def load_conf(conf):
    conf.backbone.background = conf.data.use_background

    # Player coords configuration
    conf.backbone.model.use_player_coords = conf.data.use_player_coords
    conf.transformer.model.dim = conf.backbone.model.dim
    if conf.data.use_player_coords:
        conf.transformer.model.dim += 2

    # Positional, patch, modality encoding configuration
    conf.backbone.model.use_positional_encoding = conf.transformer.model.get('use_positional_encoding', False)
    conf.backbone.model.use_patch_encoding = conf.transformer.model.get('use_patch_encoding', False)
    conf.backbone.model.use_modality_encoding = conf.transformer.model.get('use_modality_encoding', False)

    # Number of classes
    conf.transformer.model.num_classes = conf.data.num_classes

    # Batch, Sequence shape configuration
    conf.backbone.model.batch_first = conf.transformer.model.batch_first

    conf.backbone.weights = None
    conf.transformer.weights = None
    return conf