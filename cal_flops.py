# * flops and params
import time
import numpy as np
from thop import profile
import torch
import os
from utils import yaml_read
from utils.conf_base import Default_Conf
import argparse
from pathlib import Path
import pandas as pd
import hydra
from utils.select_network import select_network


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--conf_path', type=str, help='conf_path')
    parser.add_argument('--gpus', type=str, help='use which gpu')
    parser.add_argument('--epochs', type=int, help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, help='batch-size')
    parser.add_argument('--network', type=str, help='decide which network to use')
    parser.add_argument("--init_lr", type=float, help="learning rate")
    parser.add_argument("--load_mode", type=int, help="decide how to load model")
    parser.add_argument('-k', "--ckpt", type=str, help="path to the checkpoints to resume training")
    parser.add_argument("--use_scheduler", action="store_true", help="use scheduler")
    parser.add_argument('--aug', action='store_true', help='data augmentation')
    parser.add_argument('--save_arch', type=str, help="save arch")
    parser.add_argument('--file_name', type=str, default=os.path.basename(__file__).split('.')[0], help='file name')

    parser.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    parser.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')

    return parser

@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(conf):
    conf = conf["config"]
    warming_up_times = 5
    inference_times = 10

    flops_ls, params_ls, network_ls, inference_ls = [], [], [], []
        # try:

    model = select_network(conf).cuda()
    model.eval()
    input = torch.randn(1, 1, 64, 64, 64).cuda()

    # warm up
    for _ in range(warming_up_times):
        start = time.time()
        outputs = model(input)
        torch.cuda.synchronize()
        end = time.time()
        # print('Time:{}ms'.format((end-start)*1000))

    # inference time
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
    #     outputs = model(input)
    # print(prof.table())
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((inference_times, 1))
    with torch.no_grad():
        for i in range(inference_times):
            starter.record()
            outputs_2 = model(input)
            ender.record()
            torch.cuda.synchronize()
            cost_time = starter.elapsed_time(ender)
            timings[i] = cost_time
            print(f'Times:{cost_time} ms')
        avg = timings.max()
        print(f'{conf.network} inference time (AVG):{avg} ms')
    # print(prof.key_averages().table(sort_by='self_cuda_time_total'))
    # prof.export_chrome_trace(f'./{conf.network}_profile.json')
    flops, params = profile(model, inputs=(input,))

    print(f'FLOPs = {str(flops/1000**3)} G')
    print(f'Params = {str(params/1000**2)} M')

    flops_ls.append(str(flops/1000**3)+'G')
    params_ls.append(str(params/1000**2)+'M')
    network_ls.append(conf.network)
    inference_ls.append(str(avg)+'ms')
    flops= str(flops/1000**3)+'G'
    params= str(params/1000**2)+'M'
    network = conf.network
    inference_time = str(avg)+'ms'
    # except:
    # print(f'network : {conf.network} have bug ! skip this network')
    # flops_ls.append(None)
    # params_ls.append(None)
    # network_ls.append(conf.network)
    # inference_ls.append(None)
    df = pd.read_csv('./flops_statistics.csv')
    if conf.network not in df['name'].values:
        data = {'name': network, 'FLOPs': flops, 'Params': params, 'Inference_time': inference_time}
        df.loc[len(df)] = data
        df.to_csv('./flops_statistics.csv', index=False)
    else:
        df.loc[df['name'] == conf.network, 'FLOPs'] = flops
        df.loc[df['name'] == conf.network, 'Params'] = params
        df.loc[df['name'] == conf.network, 'Inference_time'] = inference_time
        df.to_csv('./flops_statistics.csv', index=False)
    # conf_path = args.conf_path
    # conf = Default_Conf()
    # conf.update(yaml_read(conf_path))
    # conf.update_from_args(args_dict)
    #
    # input = torch.randn(1, 1, 64, 64, 64).cuda()
    # flops, params = profile(model, inputs=(input,))
    # print(f'FLOPs = {str(flops/1000**3)} G')
    # print(f'Params = {str(params/1000**2)} M')
    
if __name__ == "__main__":
    main()
    
