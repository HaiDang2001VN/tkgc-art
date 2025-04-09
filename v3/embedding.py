import os
from torch import nn
import torch
from transe_model import KnowledgeEmbedding
from easydict import EasyDict as edict
import numpy as np
import torch.optim as optim

class EmbeddingManager:
    def __init__(self, graph, train_config, node_dim: int = 128):
        super().__init__()
        self.graph        = graph
        self.node_dim     = node_dim
        self.padding      = torch.zeros(node_dim)
        self.train_config = train_config
        
        self.embeddings = AuthorEmbedding(self.graph['num_nodes'], self.node_dim, self.graph['node_degrees'])
        
        model_file = f'{train_config["log_dir"]}/transe_model_sd_epoch_{train_config["num_epochs"]}.ckpt'
        if os.path.exists(model_file):
            print(f'[EMBED] Load TransE embeddings {model_file}\n\n')
            try:
                self.embeddings.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
            except:
                print(f'[EMBED] Encounter error when loading embeddings. Rebuild embeddings\n\n')
                self.train()
        else:
            print(f'[EMBED] TransE embeddings are not ready. Build embeddings\n\n')
            self.train()
    
    def get_embedding(self, node):
        return getattr(self.embeddings, 'author')(torch.tensor([node])).view(-1)
    
    def get_relation_embedding(self):
        return getattr(self.embeddings, 'collab').view(-1)
            
    def score(self, embedding1, embedding2):
        return torch.cosine_similarity(embedding1, embedding2, dim=0).item()
        
    def update_path_embedding(self, path_embedding, node_embedding):
        return (path_embedding + node_embedding) / 2
      
    def get_path_tokens(self, path, mask):
        embedding = torch.empty((0, self.node_dim))
        
        for (node, msk) in zip(path, mask):            
            if msk == 0:
                embedding = torch.cat((embedding, self.padding.unsqueeze(0)))
            else:
                embedding = torch.cat((embedding, self.get_embedding(node).unsqueeze(0)))
                
        return embedding
      
    def get_path_tokens(self, path, mask):
        embedding = torch.empty((0, self.node_dim))
        
        for (node, msk) in zip(path, mask):            
            if msk == 0:
                embedding = torch.cat((embedding, self.padding.unsqueeze(0)))
            else:
                embedding = torch.cat((embedding, self.get_embedding(node).unsqueeze(0)))
                
        return embedding
    
    def train(self):
        dataloader = EmbeddingDataLoader(self.graph, self.train_config['batch_size'])
        optimizer = optim.SGD(self.embeddings.parameters(), lr=self.train_config['lr'])
        steps = 0
        smooth_loss = 0.0
        
        print(f'[EMBED] Start to run {self.train_config["num_epochs"]} epochs with learning rate {self.train_config["lr"]:.5f}\n\n')

        for epoch in range(1, self.train_config['num_epochs'] + 1):
            dataloader.reset()
            
            while dataloader.has_next():
                # Get training batch.
                batch_indices = dataloader.get_batch()
                batch_indices = torch.from_numpy(batch_indices).to(self.train_config['devices'])

                # Train model.
                optimizer.zero_grad()
                train_loss = self.embeddings(batch_indices)
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.embeddings.parameters(), self.train_config['max_grad_norm'])
                optimizer.step()
                smooth_loss += train_loss.item() / self.train_config['steps_per_checkpoint']

                steps += 1
                if steps % self.train_config['steps_per_checkpoint'] == 0:
                    store_path = f'{self.train_config["log_dir"]}/transe_model_sd_epoch_{epoch}_step_{steps}.ckpt'
                    print(f'[EMBED] Epoch {epoch:02d} - Training smooth loss: {smooth_loss:.5f} at {store_path}\n\n')
                    smooth_loss = 0.0
                    torch.save(self.embeddings.state_dict(), store_path)


class AuthorEmbedding(KnowledgeEmbedding):
    def __init__(self, num_authors, node_dim, node_degrees, num_neg_samples: int = 10, l2_lambda: float = 0.1):
        nn.Module.__init__(self)
        self.embed_size      = node_dim
        self.num_neg_samples = num_neg_samples
        self.l2_lambda       = l2_lambda
        self.padding         = torch.zeros(node_dim)
        self.device          = 'cpu'
        
        # Initialize entity embeddings.
        self.entities = edict(
            author=edict(vocab_size=num_authors)
        )
        setattr(self, 'author', self._create_entity_embedding(num_authors))
        
        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            collab=edict(
                et='author',
                et_distrib=self._normalize_distribution(node_degrees)
            )
        )        
        setattr(self, 'collab', self._create_relation_embedding())
        setattr(self, 'collab_bias', self._create_relation_bias(len(self.relations['collab'].et_distrib)))
    
    def compute_loss(self, batch_indices):
        head_author_indices = batch_indices[:, 0]
        tail_author_indices = batch_indices[:, 1]
        
        # author + collab -> author
        loss, embeds = self._negative_sampling_loss(
            'author', 'collab', 'author', head_author_indices, tail_author_indices
        )
        
        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = sum(torch.norm(term) for term in embeds)
            loss += self.l2_lambda * l2_loss
        
        return loss


class EmbeddingDataLoader:
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size
        self.reset()
        
    def reset(self):
        self.train_edges = np.random.permutation(len(self.graph['splits']['base']))
        self.cur_edge_i = 0
    
    def has_next(self):
        return self.cur_edge_i < len(self.train_edges)
    
    def get_batch(self):
        '''Return a matrix of [batch_size x 2]'''
        
        batch = []
        
        if self.has_next():
            batch = self.graph['splits']['train']['edges'][self.train_edges[self.cur_edge_i:self.cur_edge_i + self.batch_size]]
            self.cur_edge_i += self.batch_size
        
        return np.array(batch)