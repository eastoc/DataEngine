# Time: 2023.10.15
import json
import numpy as np
from .utils import write_json, load_json

class dictionary_json():
    def __init__(self, data, n_clusters=100) -> None:
        self.nodes = []
        self.n_clusters = n_clusters
        self.dataset = data
        self.convert_format()
        self.mode = 'dataset'

    def build(self):
        self.build_dict()
        self.build_map()
        self.save_nodes()

    def build_dict(self): 
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
        def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
            #return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
            return cosine_similarity(X, Y)
        KMeans.euclidean_distances = euc_dist

        feats = []
        for i, data in enumerate(self.dataset):
            feats.append([data['embedding']])
        feats = np.concatenate(feats)

        self.cluster = KMeans(n_clusters = self.n_clusters, n_init='auto', random_state = 0).fit(feats)
        self.index = self.cluster.labels_
        self.nodes = self.cluster.cluster_centers_ # bags of the features
     
        for i, data in enumerate(self.dataset):
            self.dataset[i]['index'] = int(self.index[i])
        self.save_dataset()
    
    def build_map(self):
        # Hash map
        # dict('1': [dict('embedding': [], 'image':'1.jpg'), 
        #            dict('embedding': [], 'image':'2.jpg')],
        #      '2': [dict('embedding': [], 'image':'3.jpg')]
        #      ...
        #      'n_cluster' :[dict(embedding': [], 'image':'100000.jpg')]
        #     )
        
        if len(self.dataset)==0:
            print('Warning: dataset.json is empty.')
        elif 'index' not in self.dataset[0].keys():
            print('dictionary is not build.')
        else:
            # initial map
            self.map = dict()
            for i in range(self.n_clusters):
                self.map[str(i)] = []
            
            for i, data in enumerate(self.dataset):
                id = str(data['index'])
                self.map[id].append(data)
            self.save_map()

    def save_nodes(self):
        node_dict = {"nodes": self.nodes.tolist()}
        with open('nodes.json', 'w', encoding='utf-8') as file:
            json.dump(node_dict, file)
            file.close()
        print('Save nodes.json: Done')

    def save_map(self):
        with open('map.json', 'w', encoding='utf-8') as file:
            json.dump(self.map,file)
            file.close()
        print('Save map.json: Done.') 

    def save_dataset(self):
        for i, data in enumerate(self.dataset):
            self.dataset[i]['embedding'] = data['embedding'].tolist()
        write_json(self.dataset, self.mode+'.json')
    
    def convert_format(self):
        """
        convert the embedding format from list to numpy array
        List->numpy.array
        """
        for i, feat in enumerate(self.dataset):
            self.dataset[i]['embedding'] = np.array(feat['embedding'])
        print('format has been converted to numpy array.')

class dictionary():
    # MySQL dictionary
    def __init__(self, ids, feats, n_clusters=100) -> None:
        # data: pymysql.connect.cursor
        self.nodes = []
        self.n_clusters = n_clusters
        self.ids = ids
        self.feats = feats
    
    def build(self):
        self.build_dict()
        self.build_map()
        self.save_nodes()

    def build_dict(self): 
        # add node id using kmeans
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
        def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
            #return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
            return cosine_similarity(X, Y)
        KMeans.euclidean_distances = euc_dist

        self.cluster = KMeans(n_clusters = self.n_clusters, 
                              n_init='auto', init='k-means++', max_iter=200, verbose=1, random_state = 0).fit(self.feats)
        self.index = self.cluster.labels_
        self.nodes = self.cluster.cluster_centers_ # bags of the features
        
        # creat table and update node
        
        return self.index, self.nodes
        
    
    def build_map(self):
        # Hash table: a SQL database
        # ___________________________
        # | 0 ｜ 2 ｜ 3 ｜ 4 ｜ 5 ｜ 6｜ node: id
        # ---------------------------
        # |10 | 11 | 17 | 1 |  0 | 9 | image: id
        # | 2 | 13 |    |   |    |   |
        #   .    .    .   .    .   .
        #  ---------------------------
        if len(self.dataset)==0:
            print('Warning: dataset.json is empty.')
        elif 'index' not in self.dataset[0].keys():
            print('dictionary is not build.')
        else:
            # initial map
            self.map = dict()
            for i in range(self.n_clusters):
                self.map[str(i)] = []
            
            for i, data in enumerate(self.dataset):
                id = str(data['index'])
                self.map[id].append(data)
            self.save_map()

    def save_nodes(self):
        node_dict = {"nodes": self.nodes.tolist()}
        with open('nodes.json', 'w', encoding='utf-8') as file:
            json.dump(node_dict, file)
            file.close()
        print('Save nodes.json: Done')

    def save_map(self):
        with open('map.json', 'w', encoding='utf-8') as file:
            json.dump(self.map, file)
            file.close()
        print('Save map.json: Done.') 

    def save_dataset(self):
        for i, data in enumerate(self.dataset):
            self.dataset[i]['embedding'] = data['embedding'].tolist()
        write_json(self.dataset, self.mode+'.json')

def load_nodes(nodes_dir: str):
    nodes_data = load_json(nodes_dir)
        

if __name__=='__main__':
    database = load_json('dataset.json')
    BoW = dictionary_json(database, n_clusters=100)
    BoW.build()
    #print(database.keys())