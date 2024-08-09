import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable



# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

#  parameters

MAP_CUTOFF = 14

INPUT_DIM =  68
HIDDEN_DIM = 256
LAYER = 8

DROPOUT = 0.2
ALPHA = 0.7
LAMBDA = 1.5
VARIANT = True # From GCNII

LEARNING_RATE = 1E-5
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2 # [not bind, bind]
NUMBER_EPOCHS = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):

    pssm_feature = np.load('./Feature/' + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load('./Feature/' + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis = 1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load('./Feature/' + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_ProtT5_features(sequence_name):
    ProtT5_feature = np.load("./Embeddings/"  + sequence_name + '.npy')
    return ProtT5_feature.astype(np.float32)


def get_onehot_features(sequence_name):
    onehot_features = np.load('./onehot/'  + sequence_name + '.npy')
    return onehot_features.astype(np.float32)


def norm_dis(mx): # from SPROF
    return 2 / (1 + (np.maximum(mx, 4) / 4))


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_graph(sequence_name):
    dismap = np.load('./Feature/' + "distance_map/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix




class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values



    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])

        sequence_embedding = embedding(sequence_name, sequence, EMBEDDING)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)


        graph = load_graph(sequence_name)

        return sequence_name, sequence, label, node_features, graph

    def __len__(self):
        return len(self.labels)







class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual  = residual
        self.weight =Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = min(1, math.log(lamda/l+1))
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual: # speed up convergence of the training process
            output = output+input
        return output




class GCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        return layer_inner
        

input_size = 20
hidden_size = 32
num_layers = 2
import torch
import torch.nn as nn
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(BiLSTMModel, self).__init__()

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size_layer = input_size if i == 0 else 2 * hidden_size  
            self.lstm_layers.append(nn.LSTM( input_size_layer, hidden_size=hidden_size, batch_first=True, bidirectional=True))

        self.dropout_layers = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(num_layers)])
    def forward(self, x):
        for i in range(len(self.lstm_layers)):
            x = self.dropout_layers[i](x)
            lstm_out, _ = self.lstm_layers[i](x)
            x = lstm_out  

        output = lstm_out
        output = torch.squeeze(output)

        return output        
class MIPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(MIPPIS, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(54, 32),
            nn.ReLU(),
            nn.Linear(32, 14),
        )
        self.act_fn = nn.ReLU()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(1344, 256))
        self.fcs.append(nn.Linear(256, 128))
        self.fcs.append(nn.Linear(128, nclass))
        self.BiLSTMModel = BiLSTMModel(input_size, hidden_size, num_layers)
        self.deep_gcn = GCN(nlayers = nlayers, nfeat = nfeat, nhidden = nhidden, nclass = nclass,
                                dropout = dropout, lamda = lamda, alpha = alpha, variant = variant)
        self.criterion = nn.CrossEntropyLoss() # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    def forward(self, x, adj,seq):          # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        chain_embedding = get_ProtT5_features(seq)
        chain_embedding = torch.tensor(chain_embedding)
        chain_embedding = Variable(chain_embedding.cuda())
        onehot = get_onehot_features(seq)
        onehot = torch.tensor(onehot)
        onehot = Variable(onehot.cuda())

        x1 = self.mlp(x)
        x = torch.cat((x1, x), dim=1) 
      
        x = x.float()
        output = self.deep_gcn(x, adj) 
     
        output1 = self.BiLSTMModel(onehot)

        output= torch.cat((output, output1), dim=1)
        output= torch.cat((output, chain_embedding), dim=1)
        output = F.dropout(output, 0.2, training=self.training)

        output = self.act_fn(self.fcs[0](output))
        output = F.dropout(output, 0.2, training=self.training)
        output = self.act_fn(self.fcs[1](output))
        output = F.dropout(output, 0.2, training=self.training)
        output = self.fcs[2](output)
        return output

