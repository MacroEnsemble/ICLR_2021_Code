import os
import sys
import time
import logging

import re

import sshex.logger

#sshex.logger.Config.Use(colored=True, level='DEBUG')
LOG = sshex.logger.getLogger()


config = {
    'server': {}
}

PROC_COUNTER = 0

def retry(max_times=3, handler=None, **handler_kwargs):
    def wrapper(f):
        def wrapped_func(*args, **kwargs):
            times = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except:
                    LOG.exception('Error occurred:')

                    time.sleep(10)
                    LOG.info('sleep for 10 sec')

                    if times >= max_times:
                        LOG.info('max recovery time = {} exceeded'.format(max_times))
                        raise

                    if handler is not None:
                        LOG.info('call handler')
                        handler(**handler_kwargs)
                    times += 1

                    LOG.info('retry == {}'.format(times))

        return wrapped_func
    return wrapper



def _get_proc_number():
    global PROC_COUNTER
    PROC_COUNTER += 1
    return PROC_COUNTER

def _parse(line, parser):

    matches = parser.findall(line)
    if len(matches) == 0:
        return None
        
    results = []
    for match in matches:
        if match[0] == 'headers':
            headers = [s.strip() for s in match[1].split(',')]
            return True, headers
        elif match[0] == 'i':
            results.append(int(match[1]))
        elif match[0] == 'f':
            results.append(float(match[1]))
        elif match[0] == 's':
            results.append(match[1])
        else:
            raise Exception('Unknown type: type: {}, value:{}'.format(match[0], match[1]))
    return False, results

def _parse_results(res):
    field_parser = re.compile(r'(\w+)={([^}]*)}')

    results = []
    headers=None

    for line in res:
        is_header, values = _parse(line, field_parser)
        if is_header:
            headers = values
        else:
            results.append(values)

    if headers is not None:
        results = [dict(zip(headers, v)) for v in results]

    d = {}

    for idx, v in enumerate(results):
        if isinstance(v, dict) and v.get('index', None) is not None:
            d[v['index']] = v
        else:
            d[idx] = v

    return d


def _ssh_system_exec(server=0, command=None, sh=None, args=None):

    user_ = config['server'][server].get('user', None)
    pass_ = config['server'][server].get('pass', None)

    def _executor(executor, script, log=None, level='DEBUG'):
        if log is not None:
            LOG.log(logging.getLevelName(level), '{}: {}'.format(log, script))

        return executor(script)

    if config['server'][server].get('use_password', config.get('use_password', False)):
        import pexpect as p

        if sh is not None:
            assert os.path.isfile(sh) # check file exists
            proc = _executor(p.spawn, "ssh {} 'bash -s' < {}".format(user_, sh), 'spawn process (PIPED)') # spawn process
        elif command is not None:
            assert isinstance(command, str)
            proc = _executor(p.spawn, "ssh {} '{}'".format(user_, command), 'spawn process (PIPED)')
        
        timeout=config.get('timeout', 30)
        LOG.debug('expecting: "password:", timeout={}s'.format(timeout))
        proc.expect('password:', timeout=timeout) # wait for password query, timeout = 30 sec
        LOG.debug('sending password')
        proc.sendline(pass_) # send password
        
        stdout = proc.read().decode('utf-8').splitlines()

        proc.expect(p.EOF)
        proc.close()

        exc = proc.exitstatus

    else:
        if sh is not None:
            assert os.path.isfile(sh)
            proc = _executor(os.popen, "ssh {} 'bash -s' < {}".format(user_, sh), 'spawn process (PIPED)')
        elif command is not None:
            assert isinstance(command, str)
            proc = _executor(os.popen, "ssh {} '{}'".format(user_, command), 'spawn process (PIPED)')
        
        stdout = proc.read().splitlines()

        exc = proc.close()
        exc = 0 if exc is None else exc

    return stdout, exc

def _get_script_path(script_name):
    return os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    script_name)

def _check_tmux_config():
    if not 'tmux' in config:
        config['tmux'] = {}
    if not 'session_name' in config['tmux']:
        config['tmux']['session_name'] = 'sshex-session'
    if not 'window-prefix' in config['tmux']:
        config['tmux']['window_prefix'] = 'sshex'

def exe(server=0, command=None, sh=None, py=None, args=None):
    user_ = config['server'][server].get('user', None)
    pass_ = config['server'][server].get('pass', None)

    proc_num = _get_proc_number()

    LOG.debug('get proc number: {}'.format(proc_num))

    def _executor(executor, script, log=None, level='INFO'):
        if log is not None:
            LOG.log(logging.getLevelName(level), '{}: {}'.format(log, script))

        return executor(script)



    if not has_session(server=server):
        _tmux_command = "ssh {} 'tmux new-session -d -s \"{}\" -n \"{}-{}\" {{}}".format(
                                user_,
                                config['tmux']['session_name'], 
                                config['tmux']['window_prefix'], 
                                proc_num)
    else:
        _tmux_command = "ssh {} 'tmux new-window -t \"{}:\" -n \"{}-{}\" {{}}".format(
                                user_,
                                config['tmux']['session_name'],
                                config['tmux']['window_prefix'],
                                proc_num)

    set_env_command = 'source /etc/profile; export SSHEX_PROC_NUM={}'.format(proc_num)

    if config['server'][server].get('use_password', config.get('use_password', False)):
        import pexpect as p

        if command is not None:
            assert isinstance(command, str)
            proc = _executor(p.spawn, _tmux_command.format("'\"'\"'{}; {}'\"'\"".format(set_env_command, command)), 'spawn process (PIPED)')
            #proc = _executor(p.spawn, "ssh {} tmux new-session -d -s {} '{}'".format(user_, proc_num, command), 'spawn process (PIPED)')
        elif sh is not None:
            proc = _executor(p.spawn, _tmux_command.format("'\"'\"'{}; {}'\"'\"".format(set_env_command, sh)), 'spawn process (PIPED)')
        elif py is not None:
            proc = _executor(p.spawn, _tmux_command.format("'\"'\"'{}; python3 {}'\"'\"".format(set_env_command, py)), 'spawn process (PIPED)')
        
        timeout=config.get('timeout', 30)
        LOG.debug('expecting: "password:", timeout = {}'.format(timeout))
        proc.expect('password:', timeout=timeout)
        LOG.debug('sending password')
        proc.sendline(pass_)

        LOG.debug('expecting: "EOF (stdout)", timeout = {}'.format(timeout))
        proc.expect(p.EOF)
        proc.close()

        exc = proc.exitstatus

    else:
        if command is not None:
            assert isinstance(command, str)
            exc = _executor(os.system, _tmux_command.format("'\"'\"'{}; {}'\"'\"".format(set_env_command, command)), "spawn process (UNPIPED)")
        elif sh is not None:
            exc = _executor(os.system, _tmux_command.format("'\"'\"'{}; {}'\"'\"".format(set_env_command, sh)), "spawn process (UNPIPED)")
        elif py is not None:
            exc = _executor(os.system, _tmux_command.format("'\"'\"'{}; python3 {}'\"'\"".format(set_env_command, py)), "spawn process (UNPIPED)")

    if exc != 0:
        LOG.warning('got an unexpected exit code: {}'.format(exc))
        LOG.warning('    Please check your program.')
    else:
        LOG.debug("got exit code: {}".format(exc))

    return exc, proc_num


def hostname(server=0):

    sh = _get_script_path('get_hostname.sh')
    res, exc = _ssh_system_exec(server=server, sh=sh)

    LOG.debug('exc: {}, result: {}'.format(exc, res))

    return exc, res


def cpu_info(server=0):
    pass

def gpu_info(server=0):
    
    LOG.debug('querying for gpu information on the server {}'.format(server))

    sh = _get_script_path('gpu_info.sh')
    res, exc = _ssh_system_exec(server=server, sh=sh)

    d = _parse_results(res)

    LOG.debug('exc: {}, result: {}'.format(exc, d))

    '''
    field_parser = re.compile(r'(\w+)={([^}]*)}')
    
    results = []
    headers=None

    for line in res:
        is_header, values = _parse(line, field_parser)
        if is_header:
            headers = values
        else:
            results.append(values)

    if headers is not None:
        results = [dict(zip(headers, v)) for v in results]

    d = {}

    for idx, v in enumerate(results):
        if isinstance(v, dict) and v.get('index', None) is not None:
            d[v['index']] = v
        else:
            d[idx] = v
    '''
    return exc, d

def sys_load(server=0):

    LOG.debug('querying for system load of the server {}'.format(server))

    sh = _get_script_path('system_load.sh')
    res, exc = _ssh_system_exec(server=server, sh=sh)

    d = _parse_results(res)[0]

    LOG.debug('exc: {}, result: {}'.format(exc, d))

    return exc, d

def cpu_usage(server=0):

    LOG.debug('querying for cpu utilization of the server {}'.format(server))

    sh = _get_script_path('cpu_usage.sh')
    res, exc = _ssh_system_exec(server=server, sh=sh)

    LOG.debug('exc: {}, result: {}'.format(exc, res))
    LOG.debug(_parse_results(res))
    
    d = _parse_results(res)[0]

    
    

    LOG.debug('exc: {}, result: {}'.format(exc, d))

    return exc, d

def gpu_usage(server=0):

    LOG.debug('querying for gpu utilization of the server {}'.format(server))

    sh = _get_script_path('gpu_usage.sh')
    res, exc = _ssh_system_exec(server=server, sh=sh)

    d = _parse_results(res)

    '''
    field_parser = re.compile(r'(\w+)={([^}]*)}')

    results = []
    headers=None

    for line in res:
        is_header, values = _parse(line, field_parser)
        if is_header:
            headers = values
        else:
            results.append(values)

    if headers is not None:
        print(headers)
        results = [dict(zip(headers, v)) for v in results]

    d = {}

    for idx, v in enumerate(results):
        if isinstance(v, dict) and v.get('index', None) is not None:
            d[v['index']] = v
        else:
            d[idx] = v
    '''

    LOG.debug('exc: {}, result: {}'.format(exc, d))

    return exc, d


def has_session(server=0):

    _check_tmux_config()

    name = config['tmux']['session_name']

    _, exc = _ssh_system_exec(server=server, command="tmux has-session -t \"{}\"".format(name))

    LOG.debug('querying has-session, results: {}'.format(exc == 0))

    return exc == 0


def has_process(server=0, proc_num=None):

    if not has_session(server=server):
        return 0

    name = config['tmux']['session_name']

    res, exc = _ssh_system_exec(server=server, command="tmux list-windows -t \"{}\" | grep \"{}-{}[^0-9]\" | wc -l".format(
                                                                    name,
                                                                    config['tmux']['window_prefix'],
                                                                    proc_num))

    if exc != 0:
        return False

    if isinstance(res, list):
        res = int(''.join(res))

    return res >= 1


def num_process(server=0):

    if not has_session(server=server):
        return 0, 0
    
    name = config['tmux']['session_name']
    res, exc = _ssh_system_exec(server=server, command="tmux list-windows -t \"{}\" | grep {} | wc -l".format(name, config['tmux']['window_prefix']))

    if exc != 0:
        return exc, 0

    if isinstance(res, list):
        res = int(''.join(res))

    return exc, res




__all__ = [
    'config',
    cpu_usage.__name__,
    gpu_usage.__name__,
    sys_load.__name__,
    gpu_info.__name__,
    exe.__name__,
    has_process.__name__,
    num_process.__name__,
    has_session.__name__,
    hostname.__name__
]
