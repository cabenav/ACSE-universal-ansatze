# credit by nanoGPT
"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

'''
    Following credited to Weilei Zeng, Nov. 2024
'''


def print_config(globals_output:dict,additional_keys:list = None):
    '''
    print varibles in the current runtime
    how to use:
        from configurator import print_config
        print_config(globals()) 
        add additinal keys for list to print and save
    '''
    config_keys = [k for k,v in globals_output.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str)) and k not in ['arg','key','val','attempt']]
    if additional_keys:
        config_keys.extend(additional_keys)
    config = {k: globals_output[k] for k in config_keys} # will be useful for logging
    import json
    print('CONFIG:')
    print(json.dumps(config, indent=2))
    return config_keys, config

import json
import os

# save config into file, or verify config if file already exist
def save_config(config:dict, filename_config_json, override=False,verify=True,important_keys=['hidden_size']):
# verify existing config file or create new one.
    if os.path.exists(filename_config_json):
        print(f'Found existing config file: {filename_config_json}')
        if verify:
            with open(filename_config_json, 'r') as f:
                # if file already exist, verify the config are the same
                cfg = json.load(f)
                #important_keys =['LAYERS','hidden_size']
                #important_keys =['hidden_size']
                for k in important_keys:  # make sure important keys are the same
                    assert config[k] == cfg[k]
                #for k in config_keys:    # report changed config
                for k,_ in cfg.items():
                    if config[k] != cfg[k]:
                        print(f'* Changed config: [{k}]\t  {cfg[k]} \t--> {config[k]}')        

    #if not evaluation_only:
    with open(filename_config_json, 'w') as f:
        json.dump(config, f,indent=2)
    print(f'Config saved/overrided into {filename_config_json}')


# print config before exit
def handler(signum, frame):
    '''
        print config upon Ctrl+C:
        import signal
        from configurator import handler
        signal.signal(signal.SIGINT, handler) # print config before exit
    '''
    print('*'*30,'CTRL-c received. End the program...')
    top_frame = frame
    while top_frame.f_back:
        #print('getting back')
        top_frame = top_frame.f_back    
    print_config(top_frame.f_globals)
    #print('finish printing config')
    #signal.default_int_handler(signum,frame) # recover default behavior: exit and print trace
    exit(1)
