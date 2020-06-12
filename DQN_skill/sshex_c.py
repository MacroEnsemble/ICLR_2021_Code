import os
import sys
import time
import logging

from datetime import datetime

import sshex
import sshex.logger

LOG = sshex.logger.getLogger('cluster', level='DEBUG')

_config = {}

def Config(config):
    global _config
    _config = config
    sshex.config = config


# get idle server generator
def get_idle_server(wait=True, cooldown=300, use_gpu=True):
    _try = 0
    while True:
        for server in _config['server']:

            try:

                # if it had tried all the servers and no candidate found, cool down
                if _try >= len(_config['server']):
                    
                    if wait:
                        LOG.warning('failed to find an idle server, retry times exceeded, sleep for {}s'.format(cooldown))
                        time.sleep(cooldown)
                    else:
                        if use_gpu:
                            yield None, None
                        else:
                            yield None
                    _try = 0

                _try += 1

                LOG.debug('try to find an idle server, try: {}'.format(_try))

                #_, hostname = sshex.hostname(server)
                _, cpu_usage = sshex.cpu_usage(server)
                _, sysload = sshex.sys_load(server)
                _, num_proc = sshex.num_process(server)
                
                server_has_gpu = _config['server'][server].get("has_gpu", True)
                # if use_gpu:
                if server_has_gpu and use_gpu:
                    _, gpu_usage = sshex.gpu_usage(server)
                    _, gpu_info = sshex.gpu_info(server)

                default_cpu_limit = {
                    'max_sys-load': sysload[6],  # [cpu count]
                    'min_sys-load': 0,
                    'max_num-process': 1000,
                }

                limit = _config['server'][server].get('limit', default_cpu_limit)
                
                _sysload = 0.4*sysload[0] + 0.6*sysload[1]
                max_sysload = float(limit.get('max_sys-load', default_cpu_limit['max_sys-load']))
                min_sysload = float(limit.get('min_sys-load', default_cpu_limit['min_sys-load']))
                max_numprocess = int(limit.get('max_num-process', default_cpu_limit['max_num-process']))
                
                #LOG.debug(' [DEBUGING] server {} sysload: {}, num_proc: {}, max_num-process: {}'.format(server, _sysload, num_proc, max_numprocess))

                if _sysload > max_sysload:
                    LOG.warning('server {} has exceeded max_sys-load: {}, expect: {}'.format(server, _sysload, max_sysload))
                    continue
                if _sysload < min_sysload:
                    LOG.warning('server {} has exceeded min_sys-load: {}, expect: {}'.format(server, _sysload, min_sysload))
                    continue
                if (num_proc) and (max_numprocess is not None) and (num_proc > max_numprocess):
                    LOG.warning('server {} has exceeded max_num-process: {}, expect: {}'.format(server, num_proc, max_numprocess))
                    continue

                if server_has_gpu and use_gpu:
                    found_gpu = None
                    for gpu_num in range(len(gpu_info)):

                        default_gpu_limit = {
                            'max_gpu-usage': 100,
                            'min_gpu-usage': 0,
                            'max_gpu-free': gpu_info[gpu_num]['memory.total'],
                            'min_gpu-free': 0,
                        }

                        limit = _config['server'][server].get('limit', default_gpu_limit)
                        
                        _gpuusage = gpu_usage[gpu_num][0]
                        _gpufree = gpu_info[gpu_num]['memory.free']

                        max_gpuusage = float(limit.get('max_gpu-usage', default_gpu_limit['max_gpu-usage']))
                        min_gpuusage = float(limit.get('min_gpu-usage', default_gpu_limit['min_gpu-usage']))
                        max_gpufree = float(limit.get('max_gpu-free', default_gpu_limit['max_gpu-free']))
                        min_gpufree = float(limit.get('min_gpu-free', default_gpu_limit['min_gpu-free']))

                        if _gpuusage > max_gpuusage:
                            LOG.warning('server {}\'s GPU {} has exceeded max_gpu-usage: {}, expect: {}'.format(server, gpu_num, _gpuusage, max_gpuusage))
                            continue
                        if _gpuusage < min_gpuusage:
                            LOG.warning('server {}\'s GPU {} has exceeded min_gpu-usage: {}, expect: {}'.format(server, gpu_num, _gpuusage, min_gpuusage))
                            continue
                        if _gpufree > max_gpufree:
                            LOG.warning('server {}\'s GPU {} has exceeded max_gpu-free: {}, expect: {}'.format(server, gpu_num, _gpufree, max_gpufree))
                            continue
                        if _gpufree < min_gpufree:
                            LOG.warning('server {}\'s GPU {} has exceeded min_gpu-free: {}, expect: {}'.format(server, gpu_num, _gpufree, min_gpufree))
                            continue

                        found_gpu = gpu_num
                        break

                    if found_gpu is None:
                        continue
                else:
                    found_gpu=None
            except KeyboardInterrupt:
                LOG.exception('exception occurred')
                yield None, None
            except:
                LOG.exception('exception occurred')
                continue

            if use_gpu:
                yield server, found_gpu
                _try = 0
            else:
                yield server
                _try = 0

        
        
def num_process():
    total_proc = 0

    for server in _config['server']:
        try:
            total_proc += sshex.num_process(server=server)[1]

        except:
            LOG.exception('exception occurred, when querying for number of processes on server {}'.format(server))

    return total_proc


def map(tasks, envs=None, wait=True, retry=True, cooldown=300):
    assert isinstance(tasks, list)
    
    LOG.debug('mapping tasks, retry={}, cooldown={}'.format(retry, cooldown))

    if envs is not None:
        assert len(tasks) == len(envs)

    server_generator = get_idle_server(wait=True, cooldown=cooldown, use_gpu=True)

    all_proc = []
    for idx, task in enumerate(tasks):
        if isinstance(task, str):
            command = task
        elif isinstance(task, list):
            tsk = [str(t) for t in task]
            command = ' '.join(tsk)

        if envs is not None:
            evs = envs[idx]
            ev_list = []
            
            LOG.debug('environment variables for task {}: {}'.format(idx, evs))
            
            for ev in evs:
                ev_list.append('export {}="{}"'.format(ev, evs[ev]))

            command = '; '.join(ev_list+[command])

        for _ in range(3):
            LOG.debug('searching for available server...')
            server, gpu_num = next(server_generator)
            LOG.info('distributing task {} to server {}, gpu {}'.format(idx, server, gpu_num))
            command = 'export CUDA_VISIBLE_DEVICES={}; '.format(gpu_num) + command
            exc, proc_num = sshex.exe(server=server, command=command)
        
            if exc != 0:
                LOG.error('got an unexpected exit code {} when mapping task {} (process {}) to server {}'.format(exc, idx, proc_num, server))
                if not retry:
                    LOG.warning('    no retry')
                    break
                LOG.warning('    retrying = {}'.format(_+1))
            else:
                LOG.info('Congrats! task {} (process {}) now is running on server {}'.format(idx, proc_num, server))
                break

        all_proc.append([server, proc_num])

    if wait:
    
        for server, proc_num in all_proc:
            while True:
                if not sshex.has_process(server=server, proc_num=proc_num):
                    LOG.info(' {} on server {} ended'.format(proc_num, server))
                    break
                LOG.info('process {} on server {} is still running'.format(proc_num, server))
                LOG.info('sleep for {}s'.format(cooldown))
                time.sleep(cooldown)

        LOG.info('all tasks have been completed')



__all__ = [
    Config.__name__,
    map.__name__,
]
