import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from torch_geometric.utils import subgraph
from collections import deque
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import random
import wandb

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x

class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats+hid_feats)*2, 4)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class SubgraphSampler:
    # def __init__(self, node_sampling_prob, feature_selection_prob, edge_dropout_prob):
    #     self.node_sampling_prob = node_sampling_prob
    #     self.feature_selection_prob = feature_selection_prob
    #     self.edge_dropout_prob = edge_dropout_prob

    # def sample_subgraph(self, data):
    #     device = data.x.device

    #     # Ensuring connected subgraphs via BFS
    #     num_nodes_to_sample = max(1, int(self.node_sampling_prob * data.num_nodes))
    #     start_node = random.choice(range(data.num_nodes))
    #     visited = set()
    #     queue = deque([start_node])

    #     sampled_nodes = []
    #     while queue and len(sampled_nodes) < num_nodes_to_sample:
    #         current_node = queue.popleft()
    #         if current_node not in visited:
    #             visited.add(current_node)
    #             sampled_nodes.append(current_node)
    #             neighbors = data.edge_index[1, data.edge_index[0] == current_node]
    #             for neighbor in neighbors:
    #                 if neighbor.item() not in visited and len(sampled_nodes) < num_nodes_to_sample:
    #                     queue.append(neighbor.item())

    #     sampled_nodes = th.tensor(sampled_nodes, dtype=th.long, device=device)
    #     mask = th.zeros(data.num_nodes, dtype=th.bool, device=device)
    #     mask[sampled_nodes] = True

    #     # Filter edges to retain only those within the sampled node set
    #     edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    #     sampled_edge_index = data.edge_index[:, edge_mask]

    #     # Feature selection
    #     sampled_features = random.sample(range(data.x.size(1)), int(self.feature_selection_prob * data.x.size(1)))
    #     sampled_x = th.zeros_like(data.x)
    #     sampled_x[:, sampled_features] = data.x[:, sampled_features]
    #     sampled_x = sampled_x.to(device)

    #     # Edge dropout
    #     edge_mask = th.rand(sampled_edge_index.size(1), device=device) < self.edge_dropout_prob
    #     sampled_edge_index = sampled_edge_index[:, edge_mask]
    #     sampled_edge_index = sampled_edge_index.to(device)

    #     return sampled_x, sampled_edge_index
    def __init__(self, node_sampling_prob=0.8, feature_selection_prob=0.8, edge_dropout_prob=0.2):
        """
        Parameters:
        - node_sampling_prob: Probability of sampling nodes (controls subgraph size).
        - feature_selection_prob: Proportion of features to retain.
        - edge_dropout_prob: Probability of dropping edges after node sampling.
        """
        self.node_sampling_prob = node_sampling_prob
        self.feature_selection_prob = feature_selection_prob
        self.edge_dropout_prob = edge_dropout_prob

    def sample_subgraph(self, data):
        device = data.x.device
        
        # Step 1: Random Walk-Based Node Sampling
        num_nodes_to_sample = max(1, int(self.node_sampling_prob * data.num_nodes))
        start_node = random.choice(range(data.num_nodes))
        sampled_nodes = self._random_walk(data.edge_index, start_node, num_nodes_to_sample)
        
        # Create a node mask for sampled nodes
        node_mask = th.zeros(data.num_nodes, dtype=th.bool, device=device)
        node_mask[sampled_nodes] = True

        # Step 2: Edge Masking to Retain Only Sampled Nodes
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        sampled_edge_index = data.edge_index[:, edge_mask]

        # Step 3: Apply Edge Dropout
        edge_keep_prob = 1 - self.edge_dropout_prob
        edge_dropout_mask = th.rand(sampled_edge_index.size(1), device=device) < edge_keep_prob
        sampled_edge_index = sampled_edge_index[:, edge_dropout_mask]

        # Step 4: Feature Selection
        sampled_features = random.sample(range(data.x.size(1)), 
                                         int(self.feature_selection_prob * data.x.size(1)))
        sampled_x = th.zeros_like(data.x, device=device)
        sampled_x[:, sampled_features] = data.x[:, sampled_features]
        
        # Prepare sampled data
        sampled_data = data.clone()
        sampled_data.x = sampled_x
        sampled_data.edge_index = sampled_edge_index

        return sampled_data

    def _random_walk(self, edge_index, start_node, num_nodes_to_sample):
        """
        Perform a random walk on the graph to sample nodes.
        """
        neighbors = {i: [] for i in range(edge_index.max().item() + 1)}
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            neighbors[src].append(dst)
            neighbors[dst].append(src)  # Assuming undirected graph

        visited = set()
        queue = deque([start_node])

        sampled_nodes = []
        while queue and len(sampled_nodes) < num_nodes_to_sample:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                sampled_nodes.append(current_node)
                # Randomly shuffle neighbors and add to queue
                queue.extend(random.sample(neighbors[current_node], len(neighbors[current_node])))

        return th.tensor(sampled_nodes[:num_nodes_to_sample], dtype=th.long)


class RFBoostedBiGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, node_sampling_prob=0.8, feature_selection_prob=0.8, edge_dropout_prob=0.2, num_classifiers=10):
        super(RFBoostedBiGCN, self).__init__()
        self.num_classifiers = num_classifiers
        self.bigcn_classifiers = th.nn.ModuleList([
            Net(in_feats, hid_feats, out_feats) for _ in range(num_classifiers)
        ])
        self.gcn_to_fcn_adapter = th.nn.Linear(4, hid_feats+out_feats)
        self.fcns = th.nn.ModuleList([
            th.nn.Linear(hid_feats + out_feats, hid_feats + out_feats) for _ in range(num_classifiers)
        ])
        self.final_fc = th.nn.Linear(num_classifiers * (hid_feats + out_feats), 4)
        self.sampler = SubgraphSampler(node_sampling_prob, feature_selection_prob, edge_dropout_prob)

    def forward(self, data):
        classifier_outputs = []
        # Iterate over each BiGCN and its corresponding FCN 
        for i, (gcn, fcn) in enumerate(zip(self.bigcn_classifiers, self.fcns)):
            sampled_data = self.sampler.sample_subgraph(data)
            
            # pass sampled data through BiGCN classifier
            gcn_output = gcn(sampled_data)

            # transform output for FCN compatibility
            gcn_output_transformed = self.gcn_to_fcn_adapter(gcn_output)
            
            # pass BiGCN output through FCN
            fcn_output = fcn(gcn_output_transformed)

            # Align output of BiGCN outputs to FCN with Hadamard product (RF-GNN)
            aligned_output = gcn_output_transformed * fcn_output

            # Collect outputs
            classifier_outputs.append(aligned_output)

        # Concat all classifer outputs
        final_representation = th.cat(classifier_outputs, dim=1)

        # Log softmax for class probabilities
        return F.log_softmax(self.final_fc(final_representation), dim=1)
    

def train_RFBoostedBiGCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batch_size, dataname, iter, node_sampling_prob, feature_selection_prob, edge_dropout_prob, num_classifiers, in_feats, hid_feats, out_feats):
    model = RFBoostedBiGCN(in_feats=in_feats, hid_feats=hid_feats, out_feats=out_feats, node_sampling_prob=node_sampling_prob, feature_selection_prob=feature_selection_prob, edge_dropout_prob=edge_dropout_prob, num_classifiers=num_classifiers).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
        test_loader = DataLoader(testdata_list, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
        batch_idx = 0
        avg_loss, avg_acc = [], []
        for Batch_data in tqdm(train_loader):
            Batch_data.to(device)
            out_labels = model(Batch_data)
            loss = F.nll_loss(out_labels, Batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
            _, pred = out_labels.max(dim=-1)
            train_acc = pred.eq(Batch_data.y).sum().item() / len(Batch_data.y)
            avg_acc.append(train_acc)

            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        # Validation
        # val_loss, val_acc = [], []
        model.eval()

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        with th.no_grad():
            for Batch_data in test_loader:
                Batch_data.to(device)
                out_labels = model(Batch_data)
                loss = F.nll_loss(out_labels, Batch_data.y)
                temp_val_losses.append(loss.item())
                _, val_pred = out_labels.max(dim=1)
                val_acc = (val_pred.eq(Batch_data.y).sum().item() / len(Batch_data.y))
                Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(val_pred, Batch_data.y)
                temp_val_Acc_all.append(Acc_all)
                temp_val_Acc1.append(Acc1)
                temp_val_Prec1.append(Prec1)
                temp_val_Recll1.append(Recll1)
                temp_val_F1.append(F1)

                temp_val_Acc2.append(Acc2)
                temp_val_Prec2.append(Prec2)
                temp_val_Recll2.append(Recll2)
                temp_val_F2.append(F2)

                temp_val_Acc3.append(Acc3)
                temp_val_Prec3.append(Prec3)
                temp_val_Recll3.append(Recll3)
                temp_val_F3.append(F3)
                temp_val_Acc4.append(Acc4)
                temp_val_Prec4.append(Prec4)
                temp_val_Recll4.append(Recll4)
                temp_val_F4.append(F4)
                
                temp_val_accs.append(val_acc)

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        wandb.log({
            "epoch": epoch,
            "train_loss": np.mean(avg_loss),
            "train_accuracy": np.mean(avg_acc),
            "val_loss": np.mean(temp_val_losses),
            "val_accuracy": np.mean(temp_val_accs),
            "val_acc_class_1": np.mean(temp_val_Acc1),
            "val_precision_class_1": np.mean(temp_val_Prec1),
            "val_recall_class_1": np.mean(temp_val_Recll1),
            "val_f1_class_1": np.mean(temp_val_F1),
            "val_acc_class_2": np.mean(temp_val_Acc2),
            "val_precision_class_2": np.mean(temp_val_Prec2),
            "val_recall_class_2": np.mean(temp_val_Recll2),
            "val_f1_class_2": np.mean(temp_val_F2),
            "val_acc_class_3": np.mean(temp_val_Acc3),
            "val_precision_class_3": np.mean(temp_val_Prec3),
            "val_recall_class_3": np.mean(temp_val_Recll3),
            "val_f1_class_3": np.mean(temp_val_F3),
            "val_acc_class_4": np.mean(temp_val_Acc4),
            "val_precision_class_4": np.mean(temp_val_Prec4),
            "val_recall_class_4": np.mean(temp_val_Recll4),
            "val_f1_class_4": np.mean(temp_val_F4),
        })


        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'RFBoostedBiGCN', dataname)
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
        

    return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4


if __name__ == "__main__":
    lr=0.001
    weight_decay=1e-4
    patience=3
    n_epochs=20
    batch_size=64
    TDdroprate=0.3
    BUdroprate=0.3
    node_sampling_prob = 0.7
    feature_selection_prob = 0.9
    edge_dropout_prob = 0.3
    num_classifiers = 6
    # datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
    # iterations=int(sys.argv[2])
    datasetname = "Twitter15"
    iterations = 1
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    model="RFBoostedBiGCN"

    in_feats = 5000
    hid_feats = 32
    out_feats = 4
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    wandb.init(
        project="RFBoostedBiGCN",
        config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "dataset": datasetname,
            "TDdroprate": TDdroprate,
            "BUdroprate": BUdroprate,
            "iterations": iterations,
            "model": model,
            "node_sampling_prob": node_sampling_prob,
            "feature_selection_prob": feature_selection_prob,
            "edge_dropout_prob": edge_dropout_prob,
            "num_classifiers": num_classifiers,
            "in_feats": in_feats,
            "hid_feats": hid_feats,
            "out_feats": out_feats
        }
    )
    fold_losses = []
    for iter in range(iterations):
        fold0_x_test, fold0_x_train, fold1_x_test, fold1_x_train = load2foldData(datasetname)
        treeDic = loadTree(datasetname)

        # Training on Fold 0
        train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_RFBoostedBiGCN(
            treeDic, fold0_x_test, fold0_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience,
            n_epochs, batch_size, datasetname, iter, node_sampling_prob, feature_selection_prob, edge_dropout_prob, num_classifiers, in_feats, hid_feats, out_feats)

        wandb.log({
            f"train_loss": np.mean(train_losses),
            f"val_loss": np.mean(val_losses),
            f"val_accuracy": accs0,
            f"F1_class_1": F1_0,
            f"F1_class_2": F2_0,
            f"F1_class_3": F3_0,
            f"F1_class_4": F4_0,
            "fold": 0,
            "iter": iter
        })


        # Training on Fold 1
        train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_RFBoostedBiGCN (
            treeDic, fold1_x_test, fold1_x_train, TDdroprate, BUdroprate, lr, weight_decay, patience,
            n_epochs, batch_size, datasetname, iter, node_sampling_prob, feature_selection_prob, edge_dropout_prob, num_classifiers, in_feats, hid_feats, out_feats)
        wandb.log({
            f"train_loss": np.mean(train_losses),
            f"val_loss": np.mean(val_losses),
            f"val_accuracy": accs1,
            f"F1_class_1": F1_1,
            f"F1_class_2": F2_1,
            f"F1_class_3": F3_1,
            f"F1_class_4": F4_1,
            "fold": 1,
            "iter": iter
        })
        
        fold_losses.append((np.mean(val_losses)))
        
        test_accs.append((accs0 + accs1) / 2)
        NR_F1.append((F1_0 + F1_1) / 2)
        FR_F1.append((F2_0 + F2_1) / 2)
        TR_F1.append((F3_0 + F3_1) / 2)
        UR_F1.append((F4_0 + F4_1) / 2)
    print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
    final_loss = sum(fold_losses) / iterations
    wandb.log({
        "final_test_accuracy": sum(test_accs) / iterations,
        "final_loss": final_loss,
        "final_nr_f1": sum(NR_F1) / iterations,
        "final_fr_f1": sum(FR_F1) / iterations,
        "final_tr_f1": sum(TR_F1) / iterations,
        "final_ur_f1": sum(UR_F1) / iterations,
    })

def main():
    # Sweep Configuration
    
    sweep_config = {
        "method": "bayes",  # Bayesian Optimization for best results
        "metric": {
            "name": "final_test_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "n_epochs": {
                "values": [10, 15, 20, 25]
            },
            "lr": {
                "values": [0.0001, 0.0005, 0.001, 0.005]
            },
            "weight_decay": {
                "values": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            },
            "TDdroprate": {
                "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            },
            "BUdroprate": {
                "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            },
            "node_sampling_prob": {
                "values": [0.5, 0.6, 0.7, 0.8, 0.9]
            },
            "feature_selection_prob": {
                "values": [0.5, 0.6, 0.7, 0.8, 0.9]
            },
            "edge_dropout_prob": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "num_classifiers": {
                "values": [2, 3, 4, 5, 6] 
            },
            "hid_feats": {
                "values": [4, 8, 16, 32, 64]
            },
            "out_feats": {
                "values": [4, 8, 16, 32, 64]
            },
            "batch_size": {
                "values": [16, 32, 64]  
            },
            "patience": {
                "values": [1, 2, 3, 5]
            }
        }
    }


    def train_sweep(config=None):
        with wandb.init(config=config):
            config = wandb.config
            # Load Data
            datasetname = "Twitter15" 
            iterations = 1
            in_feats = 5000


            test_accs, NR_F1, FR_F1, TR_F1, UR_F1 = [], [], [], [], []
            fold_losses = []

            for iter in range(iterations):
                treeDic = loadTree(datasetname)
                val_losses = []
                # Load 2-Fold Data
                fold0_x_test, fold0_x_train, fold1_x_test, fold1_x_train = load2foldData(datasetname)

                # Fold 0
                try:
                    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_RFBoostedBiGCN(
                        treeDic, fold0_x_test, fold0_x_train, config.TDdroprate, config.BUdroprate, config.lr,
                        config.weight_decay, config.patience, config.n_epochs, config.batch_size, datasetname, iter,
                        config.node_sampling_prob, config.feature_selection_prob, config.edge_dropout_prob,
                        config.num_classifiers, in_feats, config.hid_feats, config.out_feats)

                    wandb.log({
                        "train_loss": np.mean(train_losses),
                        "val_loss": np.mean(val_losses),
                        "val_accuracy": accs0,
                        "F1_class_1": F1_0,
                        "F1_class_2": F2_0,
                        "F1_class_3": F3_0,
                        "F1_class_4": F4_0,
                        "fold": 0,
                        "iter": iter
                    })

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("CUDA out of memory error during fold 0. Skipping this fold.")
                        continue
                # Fold 1
                try:
                    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_RFBoostedBiGCN(
                        treeDic, fold1_x_test, fold1_x_train, config.TDdroprate, config.BUdroprate, config.lr,
                        config.weight_decay, config.patience, config.n_epochs, config.batch_size, datasetname, iter,
                        config.node_sampling_prob, config.feature_selection_prob, config.edge_dropout_prob,
                        config.num_classifiers, in_feats, config.hid_feats, config.out_feats)

                    wandb.log({
                        "train_loss": np.mean(train_losses),
                        "val_loss": np.mean(val_losses),
                        "val_accuracy": accs1,
                        "F1_class_1": F1_1,
                        "F1_class_2": F2_1,
                        "F1_class_3": F3_1,
                        "F1_class_4": F4_1,
                        "fold": 1,
                        "iter": iter
                    })

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("CUDA out of memory error during fold 1. Skipping this fold.")
                        continue

                fold_losses.append(np.mean(val_losses))
                test_accs.append((accs0 + accs1) / 2)
                NR_F1.append((F1_0 + F1_1) / 2)
                FR_F1.append((F2_0 + F2_1) / 2)
                TR_F1.append((F3_0 + F3_1) / 2)
                UR_F1.append((F4_0 + F4_1) / 2)

            # Log final results
            wandb.log({
                "final_test_accuracy": np.mean(test_accs),
                "final_loss": np.mean(fold_losses),
                "final_nr_f1": np.mean(NR_F1),
                "final_fr_f1": np.mean(FR_F1),
                "final_tr_f1": np.mean(TR_F1),
                "final_ur_f1": np.mean(UR_F1)
            })

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="RFBoostedBiGCN")
    wandb.agent(sweep_id, function=train_sweep)

