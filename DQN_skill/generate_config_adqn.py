def gen_server(server_name):
    ''' Do not call this function directly. It's for function gen_config()
    Args:
        server_name:
            string. Must be one of 'hjmnoaHJMNOA'
    Returns:
        server:
            dictionary. Config of a single server.

    '''
    server_name = server_name.lower()
    assert server_name in 's', 'Unavailable server {}.'.format(server_name)
    server = dict()
    if server_name == 's':
        server = {  
                    'user': '127.0.0.1',
                    'limit': {
                        'max_sys-load': 64,
                        'max_num-process': 10,
                        'min_gpu-free': 3000,
                        }
                }
    else:
        raise ValueError('Unavailable server {}.'.format(server_name))
    # elif server_name == 'a':
    #     server = {
    #                 'user': 'usern@ip.address',
    #                 'limit': {
    #                     'max_sys-load': 32,
    #                     'max_num-process': 5,
    #                     'min_gpu-free': 3000,
    #                     }
    #                 }

    return server

def gen_config(server_names, env, timeout=30):
    '''
    Args:
        server_names:
            string. Combination of characters in 'hjmnoaHJMNOA'
            e.g. 'HJM'

        env:
            string. Name of the atari env.
            e.g. 'BeamRiderNoFrameskip-v4'

    Returns:
        config:
            dict(). Config for sshex_c.
    '''
    server_names = server_names.lower()
    server_names = ''.join(set(server_names))

    config = dict()
    config['timeout'] = timeout
    server = dict()
    for i, sn in enumerate(server_names):
        server[i] = gen_server(sn)
    config['server'] = server
    config['tmux'] = {'session_name': '\\"{}\\"'.format(env)}

    return config

if __name__ == '__main__':
    print(gen_config('omaOMA', 'BeamRiderNoFrameskip-v4'))
