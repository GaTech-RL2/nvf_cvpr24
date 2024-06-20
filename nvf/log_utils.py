import inspect
# from omegaconf import OmegaConf
import pandas as pd
import fcntl
import io
import yaml
from dataclasses import asdict
import os
from shutil import copy

# def save_cfg(path, cfg):
#     if type(cfg) is dict:
#         cfg = OmegaConf.create(cfg)
#     # with open('test.yaml', 'w') as f:
#     OmegaConf.save(config=cfg, f=path)

def save_cfg(path, cfg):
    with open(path, 'w') as f:
        yaml.dump(asdict(cfg), f)

def save_codes(path, *args):
    module_list = []
    with open(path, "w") as f:
        for obj in args:
            if inspect.ismodule(obj):
                source = inspect.getsource(obj)
            else:
                obj = inspect.getmodule(obj)
                source = inspect.getsource(obj)
            if obj in module_list:
                continue
            else:
                module_list.append(obj)
            f.write('#  '+'='*60+'\n')
            f.write('#  '+obj.__name__+'\n')
            f.write('#  '+obj.__file__+'\n')
            f.write('#  '+'='*60+'\n\n')
            f.write(source)
            f.write('\n\n')

def save_result_excel(data, filename, sheet_name=0, lock=False):
    def add_one(data,df):
        new_row = pd.DataFrame([data.values()], columns=data.keys())
        df = df._append(new_row, ignore_index=True)
        return df
    def save_all(data,file_obj):
        df = pd.read_excel(file_obj, sheet_name)

        # if lock:
        #     file_obj.seek(0)

        if type(data) is list:
            for dd in data:
                df = add_one(dd, df)
        else:
            df = add_one(data, df)

        df.to_excel(file_obj, index=False)
    
    # def create_new(data, filename):
    #     if type(data) is list:
    #         df = pd.DataFrame([d.values() for d in data], columns=data[0].keys())
    #     else:
    #         df = pd.DataFrame([data.values()], columns=data.keys())
    #     df.to_excel(filename, index=False)
    
    def copy_from(data, filename):
        empty_file_path = 'data/empty_results.xlsx'
        copy(empty_file_path, filename)

    # if lock:
    #     with open(filename, 'rb+') as file:
    #         fcntl.flock(file, fcntl.LOCK_EX)

    #         file_obj = io.BytesIO(file.read())
            

    #         save_all(data, file_obj)
            
    #         file.seek(0)
    #         file.write(file_obj.getvalue())
    #         # file.truncate()

    #         fcntl.flock(file, fcntl.LOCK_UN)
    # else:
    if not os.path.exists(filename):
        copy_from(data, filename)
        
    save_all(data, filename)

if __name__ == '__main__':
    # def test_save_codes():
    #     from ndf import train
    #     from ndf.utils import log_utils
    #     save_codes('./test.log', train, log_utils)
    # test_save_codes()
    import socket
    def test_save_result_excel():
        data={
            'time':'333',
            'scene':'333',
            'method':'333',
            'model':'333',
            # 'loss':'333',
            # 'model_path':'jbjbjb',
            'server': socket.gethostname()
        }
        save_result_excel(data, 'results/results.xlsx', lock=True)
        pass
    test_save_result_excel()
