import yaml

class Config():
    def __init__(self, yaml_path):
        yaml_file = open(yaml_path)
        _attr = yaml.load(yaml_file, Loader=yaml.FullLoader)['settings']
        for key, value in _attr.items():
            self.__dict__[key] = value

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            return None

    def __str__(self):
        print("##########   CONFIGURATION INFO   ##########")
        pretty(self.__dict__)
        return '\n'
    
    def to_dict(self):
        out_dict = {}
        for k,v in self.__dict__.items():
            if v is not None:
                out_dict[k] = v
        return out_dict
    

def config_from_dict(_dict, ignore_keys=[]):
    config = Config('./configs/configs.yaml')
    for k,v in _dict.items():
        if k not in ignore_keys:
            config.__setattr__(k,v)
    return config
        
def pretty(d, indent=0):
  for key, value in d.items():
    print('    ' * indent + str(key) + ':', end='')
    if isinstance(value, dict):
      print()
      pretty(value, indent+1)
    else:
      print('\t' * (indent+1) + str(value))