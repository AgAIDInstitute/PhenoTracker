import torch
import torch.nn as nn



class concat_embedding_vectors_phenologies_net(nn.Module):  # LT50 and budbreak
    """
    Phenology progress, no stage.
    """
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_vectors_phenologies_net, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size)
        self.linear1 = nn.Linear(input_size*2, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # Phenology Progress

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        x = torch.cat((x,embedding_out),axis=-1)
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_progress = self.linear4(out_s)  # phenology stage progress
        return out_progress, h_next, out, out_s


class progress_stage_net(nn.Module):
    """
    Phenology progress and stage.
    """
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(progress_stage_net, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size)
        self.linear1 = nn.Linear(input_size*2, 1024) #account for concatting weather input to cultivar arr (both are same size)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # Phenology Progress
        self.linear5 = nn.Linear(self.penul, 1)  #stage 0
        self.linear6 = nn.Linear(self.penul, 1)  #stage 1
        self.linear7 = nn.Linear(self.penul, 1)  #stage 2
        self.linear8 = nn.Linear(self.penul, 1)  #stage 3

    def forward(self, x, cultivar_label = None, h = None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        x = torch.cat((x,embedding_out),axis = -1)
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device = x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_progress = self.linear4(out_s)  # phenology stage progress
        out_stage0 = self.linear5(out_s)  #stage 0
        out_stage1 = self.linear6(out_s)  #stage 1
        out_stage2 = self.linear7(out_s)  #stage 2
        out_stage3 = self.linear8(out_s)  #stage 3
        return out_progress, out_stage0, out_stage1, out_stage2, out_stage3, h_next, out, out_s



