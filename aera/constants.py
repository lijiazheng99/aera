import os
import subprocess as sp

DATAFOLDER = "put your data folder here"
CACHEFOLDER = "put your cache folder here"
os.environ["TRANSFORMERS_CACHE"] = CACHEFOLDER

def __gpu_auto_select(num_gpu:int=1):
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    except:
        print("Error: Please check the avalibility of 'nvidia-smi' command")
    else:
        memory_free_values = {str(i):int(x.split()[0]) for i, x in enumerate(memory_free_info)}
        if num_gpu > len(memory_free_values):
            print(f"Error: Number of desired gpus: {num_gpu} larger than total avaliable gpus {len(memory_free_values)}.")
            raise Exception
        ranked_gpus = list(dict(sorted(memory_free_values.items(), key=lambda item: item[1], reverse=True)).keys())[:num_gpu]
        if len(ranked_gpus) > 1:
            result = ','.join(ranked_gpus)
        else:
            result = ranked_gpus[0]
        return result, 'cuda'

def get_path(foldername:str):
    return os.path.join(DATAFOLDER,foldername)

GPU_NUM, DEVICE = __gpu_auto_select(1)
print(f"***** Selected CUDA device: {GPU_NUM} *****")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
