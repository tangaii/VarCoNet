import wandb
import pickle
import optuna
import numpy as np
import pprint

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        lr = config.lr
        tau = config.tau
        layers = config.N_layers
        n_heads = config.N_heads
        dim_feedforward = config.FF_dim
        batch_size = config.batch_size
        param_row = np.array([lr,tau,layers,n_heads,dim_feedforward,batch_size]).astype('float32')
        pos = np.where(np.all(params == param_row, axis=1))[0]
        print(pos)
        wandb.log({"Obj. Func.": values[pos]})


atlas = 'AICHA'

path_trials = r'trials_VarCoNet_' + atlas + '.pkl'
path_best_params = r'best_params_VarCoNet_' + atlas + '.pkl'
with open(path_trials, 'rb') as f:
    trials = pickle.load(f)
with open(path_best_params, 'rb') as f:
    best_params = pickle.load(f)
    
lr = np.zeros((len(trials),1))
tau = np.zeros((len(trials),1))
layers = np.zeros((len(trials),1))
n_heads = np.zeros((len(trials),1))
dim_feedforward = np.zeros((len(trials),1))
batch_size = np.zeros((len(trials),1))
values = np.zeros((len(trials),))
for i,trial in enumerate(trials):
    params = trial._params
    values[i] = trial._values[0]
    lr[i] = params['lr']
    tau[i] = params['tau']
    layers[i] = params['layers']
    n_heads[i] = params['n_heads']
    dim_feedforward[i] = params['dim_feedforward']
    batch_size[i] = params['batch_size']

params = np.concatenate((lr,tau,layers,n_heads,dim_feedforward,batch_size),axis=1).astype('float32')

wandb.login()

for i in range(len(trials)):
    sweep_config = {
        'method': 'grid'
        }
    metric = {
        'name': 'Obj. Func.',
        'goal': 'maximize'   
        }
    
    sweep_config['metric'] = metric
    
    parameters_dict = {
        'lr': {
            'values': [lr[i][0]]
            },
        'tau': {
            'values': [tau[i][0]]
            },
        r'N_layers': {
              'values': [layers[i][0]]
            },
        'N_heads': {
              'values': [n_heads[i][0]]
            },
        'FF_dim': {
              'values': [dim_feedforward[i][0]]
            },
        'batch_size': {
              'values': [batch_size[i][0]]
            }
        }
    
    sweep_config['parameters'] = parameters_dict
    
    parameters_dict.update({
        'epochs': {
            'value': 1}
        })
    
    
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="VarCoNet-"+atlas+"-BO-plot_final")
        
    wandb.agent(sweep_id, train)