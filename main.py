from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, GCNConv
import uvicorn
import json

# ----------- Model Definition ----------- #

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 10 * out_channels)
        self.conv2 = GCNConv(10 * out_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# ----------- FastAPI Setup ----------- #

app = FastAPI()

class CloneRequest(BaseModel):
    file1: List[Any]
    file2: List[Any]

# Load the trained model
in_channels = 109  # based on your data processing
out_channels = 32
encoder = VariationalGCNEncoder(in_channels, out_channels)
model = VGAE(encoder)
model.load_state_dict(torch.load("vgae-model-files-ant.pt", map_location=torch.device('cpu')))
model.eval()

# Node types from training
TYPES = [
    "CLASSDECLARATION", "FIELDDECLARATION", "METHODDECLARATION", "METHODINVOCATION", "ARRAYACCESS",
    "FORSTATEMENT", "VARIABLEDECLARATIONSTATEMENT", "MEMBERDECL", "STATEMENT"
]

def get_type_id(typename):
    typename = typename.upper()
    if typename in TYPES:
        return TYPES.index(typename)
    if typename.endswith("MemberDeclaration".upper()):
        return TYPES.index("MEMBERDECL")
    if typename.endswith("MethodStatement".upper()):
        return TYPES.index("STATEMENT")
    return len(TYPES)

def json_to_graph(data_json):
    node_ids = []
    node_feats = []
    edge_index = []
    id_map = {}

    for item in data_json:
        if item[0] == "type":
            idx = item[1]
            node_type = get_type_id(item[2])
            feat = [0] * (len(TYPES) + 1)
            feat[node_type] = 1
            node_ids.append(idx)
            node_feats.append(feat)
            id_map[idx] = len(node_ids) - 1

    for item in data_json:
        if item[0] == "ast_succ":
            src, tgt = item[1], item[2]
            if src in id_map and tgt in id_map:
                edge_index.append([id_map[src], id_map[tgt]])

    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(node_feats, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

def compute_similarity(graph1, graph2):
    with torch.no_grad():
        z1 = model.encode(graph1.x, graph1.edge_index).mean(dim=0)
        z2 = model.encode(graph2.x, graph2.edge_index).mean(dim=0)
        score = F.cosine_similarity(z1, z2, dim=0).item()
    return score

@app.post("/predict")
def predict_clone(request: CloneRequest):
    try:
        graph1 = json_to_graph(request.file1)
        graph2 = json_to_graph(request.file2)
        score = compute_similarity(graph1, graph2)
        return {
            "are_clones": score > 0.8,
            "similarity_score": round(score, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment below to run locally for testing
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
