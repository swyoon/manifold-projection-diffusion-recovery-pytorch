"""
tabular.py
==========
Data loading utilities for tabular datasets
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


def get_tabular_dataset(cfg):
    name = cfg.pop('name')
    if name == 'doublepanda':
        data = get_doublepanda(**cfg)
    elif name == 'sarcos':
        data = get_sarcos(**cfg)
    elif name == 'parkinsons':
        data = get_parkinsons(**cfg)
    elif name == 'skillcraft':
        data = get_skillcraft(**cfg)
    elif name == 'SML':
        data = get_SML(**cfg)
    elif name == 'gasdrift':
        data = get_gasdrift(**cfg)
    elif name == 'pumadyn':
        data = get_pumadyn(**cfg)
    elif name == 'protein':
        data = get_protein(**cfg)
    elif name == 'elevators':
        data = get_elevators(**cfg)
    elif name == 'bike':
        data = get_bike(**cfg)
    elif name == '3droad':
        data = get_3droad(**cfg)    
    else:
        raise ValueError(f'Invalid dataset {name}')
    return data

def get_sarcos(split_seed=1, n_data=100, split='train', y_index=[0], normalize=True, verbose=-1, root='datasets', **kwargs):
    
    sarcos_inv = loadmat(os.path.join(root, 'tabular/sarcos/sarcos_inv.mat'))['sarcos_inv']
    sarcos_inv_test = loadmat(os.path.join(root, 'tabular/sarcos/sarcos_inv_test.mat'))['sarcos_inv_test']
    
    if verbose > 0:
        print(f'Total # of train data : {len(sarcos_inv)},  # of test data : {len(sarcos_inv_test)}.')
        
    rand_idx = np.arange(len(sarcos_inv))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    # predict [torque J1~J7]
    y_index = np.asarray(y_index) + 21
    assert max(y_index) < sarcos_inv.shape[1], f'index out of range (should be < {sarcos_inv.shape[1]})'

    X      = sarcos_inv[:, :21]
    Y      = sarcos_inv[:, y_index]
    X_test = sarcos_inv_test[:, :21]
    Y_test = sarcos_inv_test[:, y_index]
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=False)
        
        X = ssx.fit_transform(X)
        Y = ssy.fit_transform(Y)
        X_test = ssx.transform(X_test)
        Y_test = ssy.transform(Y_test)
        
    X = X[rand_idx[:n_data]]
    Y = Y[rand_idx[:n_data]]
    
    if split == 'train':
        return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float)
    elif split == 'test':
        return torch.tensor(X_test, dtype=torch.float), torch.tensor(Y_test, dtype=torch.float)
    else:
        assert False, f'invalid split type {split}, expected train / test'

def get_doublepanda(split_seed=1, n_data=100, split='train', root='datasets', version='vSE3pcd', threshold=0.01, **kwargs):
    data_dir = os.path.join(root, 'jihwan', version)
    X_linkori = np.load(os.path.join(data_dir, 'X_linkori.npy'))
    X_linkpos = np.load(os.path.join(data_dir, 'X_linkpos.npy'))
    X_linkpos = X_linkpos / 1.307  # scale to [-1, 1]
    y_pcd = np.load(os.path.join(data_dir, 'y_pcd.npy'))

    X = np.hstack([X_linkori, X_linkpos])  # (1000000, 126)
    y_cls = (y_pcd <= threshold).astype(np.int64)  # 22.89% 1

    idx = np.arange(len(X))
    rng = np.random.default_rng(seed=split_seed)
    rng.shuffle(idx)
    if split == 'train':
        idx = idx[:n_data]
    elif split == 'test':
        idx = idx[-n_data:]

    x = torch.tensor(X[idx], dtype=torch.float)
    y = torch.tensor(y_cls[idx], dtype=torch.long)
    return x, y

def get_parkinsons(split_seed=1, n_data=100, split='train', y_index=[0], normalize=True, verbose=-1, root='datasets', **kwargs):
    
    df_parkinsons = pd.read_csv(os.path.join(root, 'tabular/parkinsons/parkinsons_updrs.data'))
    data = df_parkinsons.values
    X_all = data[:, [*range(4), *range(6, 22)]]
    
    # predict [motor_UPDRS, total_UPDRS]
    Y_all = data[:, 4:6]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
    
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    y_index = np.asarray(y_index)
    assert max(y_index) < Y_all.shape[1], f'index out of range (should be < {Y_all.shape[1]})'
    Y_all = Y_all[:, y_index]
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=False)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)
    
def get_skillcraft(split_seed=1, n_data=100, split='train', normalize=True, verbose=-1, root='datasets', **kwargs):
    df_skillcraft = pd.read_csv(os.path.join(root, 'tabular/skillcraft/SkillCraft1_Dataset.csv'))
    df_skillcraft = df_skillcraft.replace('?', np.nan).dropna()
    df_skillcraft.Age = df_skillcraft.Age.astype(int)
    df_skillcraft.HoursPerWeek = df_skillcraft.HoursPerWeek.astype(int)
    df_skillcraft.TotalHours = df_skillcraft.TotalHours.astype(int)
    
    data = df_skillcraft.values
    X_all = data[:, [0, *range(2, 20)]]
    
    # predict LeagueIndex (Tier: bronze(0), silver(1), gold(2), ...)
    Y_all = data[:, [1]]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
        
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=False)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)

def get_SML(split_seed=1, n_data=100, split='train', y_index=[0], normalize=True, verbose=-1, root='datasets', **kwargs):
    
    df_SML1 = pd.read_csv(os.path.join(root, 'tabular/SML/NEW-DATA-1.T15.txt'), sep=' ')
    df_SML2 = pd.read_csv(os.path.join(root, 'tabular/SML/NEW-DATA-2.T15.txt'), sep=' ')
    df_SML = pd.concat([df_SML1, df_SML2])
    columns = df_SML.columns[2:]
    df_SML = df_SML.dropna(axis=1)
    df_SML = df_SML.rename(columns=dict(zip(df_SML.columns, columns)))
    df_SML['Time'] = (df_SML['1:Date'] + ' ' + df_SML['2:Time']).astype(np.datetime64)
    df_SML = df_SML.drop(columns=['1:Date', '2:Time'])

    # df_SML_last = df_SML.copy()
    # df_SML_last['Time'] += pd.Timedelta(15, unit='min')
    # df_SML_last = df_SML_last.rename(columns=
    #     {'3:Temperature_Comedor_Sensor': '3:Temperature_Comedor_Sensor_last', 
    #      '4:Temperature_Habitacion_Sensor': '4:Temperature_Habitacion_Sensor_last'}
    # )
    # df_SML = df_SML.merge(df_SML_last[['Time', '3:Temperature_Comedor_Sensor_last', '4:Temperature_Habitacion_Sensor_last']], how='left')

    df_SML_next = df_SML.copy()
    df_SML_next['Time'] -= pd.Timedelta(15, unit='min')
    df_SML_next = df_SML_next.rename(columns=
        {'3:Temperature_Comedor_Sensor': '3:Temperature_Comedor_Sensor_next', 
        '4:Temperature_Habitacion_Sensor': '4:Temperature_Habitacion_Sensor_next'}
    )
    df_SML = df_SML.merge(df_SML_next[['Time', '3:Temperature_Comedor_Sensor_next', '4:Temperature_Habitacion_Sensor_next']], how='left')
    
    # Convert date and time to timestamp format ( 2021-08-25 15:04:33.794484 -> 1629884074)
    df_SML['Time'] = df_SML['Time'].apply(lambda x:x.timestamp())
    df_SML = df_SML.dropna(axis=0)

    data = df_SML.values
    X_all = data[:, :-2]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')

    # predict the next two indoor temperature (after 15min)
    Y_all = data[:, -2:]
    
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    y_index = np.asarray(y_index)
    assert max(y_index) < Y_all.shape[1], f'index out of range (should be < {Y_all.shape[1]})'
    Y_all = Y_all[:, y_index]
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=False)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)

def get_gasdrift(split_seed=1, n_data=100, split='train', normalize=True, verbose=-1, root='datasets', **kwargs):
    
    def get_data_from_lines(x):
        class_id = int(x.split(';')[0])
        concentration = float(x.split(';')[1].split(' ')[0])
        xs = x.split(';')[1].split(' ')[1:-1]
        xs = list(map(lambda i:float(i.split(':')[1]), xs))
        xs.append(class_id)
        xs.append(concentration)
        return xs
    
    data = []
    for i in range(1, 11):
        file = open(os.path.join(root, f'tabular/gasdrift/batch{i}.dat'), 'rb')
        data += list(map(lambda x:x.decode('utf-8').strip(), file.readlines()))
        
    data = np.array(list(map(get_data_from_lines, data)))
    
    X_all = data[:, :-1]
    
    # predict concentration of gas
    Y_all = data[:, [-1]]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
        
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=False)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)

def get_pumadyn(split_seed=1, n_data=100, split='train', normalize=True, verbose=-1, root='datasets', **kwargs):
    
    data1 = np.loadtxt(os.path.join(root, f'tabular/pumadyn/puma32H.data'), delimiter=',')    
    data2 = np.loadtxt(os.path.join(root, f'tabular/pumadyn/puma32H.test'), delimiter=',')    
    data = np.concatenate((data1, data2), axis=0)
    
    X_all = data[:, :-1]
    
    # predict concentration of gas
    Y_all = data[:, [-1]]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
        
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=True)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)

def get_elevators(split_seed=1, n_data=100, split='train', normalize=True, verbose=-1, root='datasets', **kwargs):
    
    data = loadmat(os.path.join(root, f'tabular/elevators/elevators.mat'))['data']
    
    X_all = data[:, :-1]
    
    # predict concentration of gas
    Y_all = data[:, [-1]]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
        
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=True)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)

def get_bike(split_seed=1, n_data=100, split='train', normalize=True, verbose=-1, root='datasets', **kwargs):
    data = loadmat(os.path.join(root, f'tabular/bike/bike.mat'))['data']
    
    X_all = data[:, :-1]
    
    # predict concentration of gas
    Y_all = data[:, [-1]]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
        
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=True)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)

def get_protein(split_seed=1, n_data=100, split='train', normalize=True, verbose=-1, root='datasets', **kwargs):
    df = pd.read_csv(os.path.join(root, f'tabular/protein/CASP.csv'))
    data = df.values
    
    X_all = data[:, 1:]
    
    # predict concentration of gas
    Y_all = data[:, [0]]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
        
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=True)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)

def get_3droad(split_seed=1, n_data=100, split='train', normalize=True, verbose=-1, root='datasets', **kwargs):
    data = loadmat(os.path.join(root, f'tabular/3droad/3droad.mat'))['data']
    
    X_all = data[:, :-1]
    
    # predict concentration of gas
    Y_all = data[:, [-1]]
    
    if verbose > 0:
        print(f'Total # of data : {len(X_all)}. If n_train + n_test > {len(X_all)}, train and test dataset will overlap.')
        
    rand_idx = np.arange(len(X_all))
    np.random.seed(split_seed)
    np.random.shuffle(rand_idx)
    
    if normalize:
        ssx = StandardScaler()
        ssy = StandardScaler(with_std=True)
        
        X_all = ssx.fit_transform(X_all)
        Y_all = ssy.fit_transform(Y_all)
        
    if split == 'train':
        idx = rand_idx[:n_data]
    elif split == 'test':
        idx = rand_idx[-n_data:]
    else:
        assert False, f'invalid split type {split}, expected train / test'
    
    return torch.tensor(X_all[idx], dtype=torch.float), torch.tensor(Y_all[idx], dtype=torch.float)
