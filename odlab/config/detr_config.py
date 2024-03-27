# End-to-end Detection with Transformer

def build_detr_config(args):
    if   args.model == 'detr_r50':
        return Detr_R50_Config()
    else:
        raise NotImplementedError("No config for model: {}".format(args.model))


class DetrBaseConfig(object):
    def __init__(self):
        pass

    def print_config(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        for k, v in config_dict.items():
            print("{} : {}".format(k, v))

class Detr_R50_Config(DetrBaseConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Backbone
        pass
