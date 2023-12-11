import json
import numpy as np
from .engine import Engine_json
from .utils import load_json


class IRSystem():

    def __init__(self):
        # Load engine
        self.Engine = Engine_json(nodes=self.nodes)
        
def load_sql(database):
    # 加载数据
    results = database.get_data()
    
    # 获取emb和ids
    #embs = database.get_value(key='embedding')
    #ids = database.get_value(key='id')
    # 增加聚类节点key：node
    #database.add_key('node', type='INT')
    # concatenate the embeddings, shape: [512, n]
    feats = []
    ids = []
    node_id = []
    flag = 0
    for i, data in enumerate(results):
        id = data[0]
        emb = data[-2]
        if  emb != None:
            emb = json.loads(emb)
            emb = np.array(emb)
            if emb.size==512:
                ids.append(id)
                emb = np.array([emb]).reshape(512, 1)
                if flag == 0:
                    feats = emb
                else:
                    feats = np.hstack((feats,emb))
                flag = 1
    return ids, feats, node_id

if __name__=='__main__':
    # IR system
    robot = IRSystem()
    query_dir = 'images/1.jpg'
    res = robot.Engine.search(query_dir, robot.Database)
    print(res)
    
