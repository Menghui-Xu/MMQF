import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import subprocess
import csv

conda_path = '/home/ubuntu/anaconda3/bin/conda'
script_path = '/home/ubuntu/xmh/AGDA改进/main.py'



env_name   = 'MMQF'

Deng ={'dropout': 0.5,      'rels': 65,      'n_epochs': 400,     'batch_size': 2048,
       'graph_alpha': 0.6,  'min_sim': 0.6,  'temperature': 0.2,  'cl_alpha': 0.7,
       'w_cl': 0.1,         'lr': 0.002,     'dataset': 'Deng'}

Lin ={'dropout': 0.3,      'rels': 100,      'n_epochs': 400,     'batch_size': 8192,
       'graph_alpha': 0.7,  'min_sim': 0.7,  'temperature': 0.05,  'cl_alpha': 0.7,
       'w_cl': 0.1,         'lr': 0.001,     'dataset': 'Lin'}



gpu = 3
folds = [0,1,2,3,4]
seed = 0


for fold in folds:
            cmd = [conda_path, 'run', '-n', env_name, '--no-capture-output', 'python', script_path, '--dropout',str(Deng['dropout']),
                   '--rels', str(Deng['rels']), '--graph_alpha', str(Deng['graph_alpha']), '--min_sim', str(Deng['min_sim']),
                   '--w_cl', str(Deng['w_cl']), '--lr', str(Deng['lr']), '--dataset', Deng['dataset'], '--model', "Model_WO_RGCN",
                   '--batch_size',str(Deng['batch_size']), '--temperature',str(Deng['temperature']),'--n_epochs', str(400),
                   '--gpu_id', str(gpu), '--fold', str(fold), '--seed',str(seed),
                   '--filename', f"result/{gpu}_Deng_cold1.csv", ]
            subprocess.run(cmd, check=True)

