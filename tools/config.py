import yaml

url_config = {
        'vgg_transformer':'config/vgg_transformer.yml',
        'resnet_transformer':'config/resnet_transformer.yml',
        'resnet_fpn_transformer':'config/resnet_fpn_transformer.yml',
        'base':'config/base.yml',
        }

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        with open(url_config['base'], encoding='utf-8') as f:
            base_config = yaml.safe_load(f)

        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

