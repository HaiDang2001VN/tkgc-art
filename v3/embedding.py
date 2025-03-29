# embedding.py
import os
import argparse
import torch
import torch.optim as optim
# can also use torch.utils.data.DataLoader if preferred
from torch_geometric.loader import DataLoader
# Import KGE models from the dedicated kge module (new in recent PyG versions)
from torch_geometric.nn.kge import TransE, ComplEx, DistMult, RotatE
from data import KGDataset


def train(args):
    # Define your file mappings – adjust these to match your dataset.
    entity_files = {
        'user': 'users.txt.gz',
        'product': 'product.txt.gz'
    }
    relation_files = {
        'produced_by': ('brand_p_b.txt.gz', 'brand'),
        'belongs_to': ('category_p_c.txt.gz', 'category'),
        'also_bought': ('also_bought_p_p.txt.gz', 'related_product'),
        'also_viewed': ('also_viewed_p_p.txt.gz', 'related_product'),
        'bought_together': ('bought_together_p_p.txt.gz', 'related_product')
    }

    # Create your KG dataset
    dataset = KGDataset(
        data_dir=args.data_dir,
        set_name=args.set_name,
        entity_files=entity_files,
        relation_files=relation_files
    )

    # Wrap the triples in a TensorDataset and create a DataLoader.
    triple_dataset = torch.utils.data.TensorDataset(dataset.triples)
    loader = torch.utils.data.DataLoader(
        triple_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Allow selection of embedding method via command-line.
    model_dict = {
        'transe': TransE,
        'complex': ComplEx,
        'distmult': DistMult,
        'rotate': RotatE
    }
    model_name = args.model.lower()
    if model_name not in model_dict:
        raise ValueError(
            f"Model {args.model} is not supported. Choose from {list(model_dict.keys())}."
        )

    ModelClass = model_dict[model_name]
    # Instantiate the selected model using the latest API arguments.
    model = ModelClass(
        num_nodes=dataset.num_entities,
        num_relations=dataset.num_relations,
        hidden_channels=args.embed_size,
        margin=args.margin,
        p_norm=args.p_norm,
        sparse=args.sparse
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    steps = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            pos_triples = batch[0].to(device)  # Shape: [batch_size, 3]
            head, rel, tail = pos_triples[:,
                                          0], pos_triples[:, 1], pos_triples[:, 2]

            optimizer.zero_grad()
            loss = model.loss(head, rel, tail)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            if steps % args.log_steps == 0:
                print(
                    f"Epoch: {epoch}, Step: {steps}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch.
        checkpoint_path = os.path.join(
            args.log_dir, f"{model_name}_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)


def extract_embeddings(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the same dataset settings as during training.
    entity_files = {
        'user': 'users.txt.gz',
        'product': 'product.txt.gz'
    }
    relation_files = {
        'produced_by': ('brand_p_b.txt.gz', 'brand'),
        'belongs_to': ('category_p_c.txt.gz', 'category'),
        'also_bought': ('also_bought_p_p.txt.gz', 'related_product'),
        'also_viewed': ('also_viewed_p_p.txt.gz', 'related_product'),
        'bought_together': ('bought_together_p_p.txt.gz', 'related_product')
    }

    dataset = KGDataset(
        data_dir=args.data_dir,
        set_name=args.set_name,
        entity_files=entity_files,
        relation_files=relation_files
    )

    model_dict = {
        'transe': TransE,
        'complex': ComplEx,
        'distmult': DistMult,
        'rotate': RotatE
    }
    model_name = args.model.lower()
    if model_name not in model_dict:
        raise ValueError(
            f"Model {args.model} is not supported. Choose from {list(model_dict.keys())}."
        )

    ModelClass = model_dict[model_name]
    model = ModelClass(
        num_nodes=dataset.num_entities,
        num_relations=dataset.num_relations,
        hidden_channels=args.embed_size,
        margin=args.margin,
        p_norm=args.p_norm,
        sparse=args.sparse
    ).to(device)

    # Load the model checkpoint from the final epoch.
    model_file = os.path.join(
        args.log_dir, f"{model_name}_epoch_{args.epochs}.pt")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Extract and save the embeddings.
    embeddings = {
        'entity_embeddings': model.entity_embedding.weight.data.cpu().numpy(),
        'relation_embeddings': model.relation_embedding.weight.data.cpu().numpy()
    }
    save_path = os.path.join(args.log_dir, f"{model_name}_embeddings.pt")
    torch.save(embeddings, save_path)
    print(f"Embeddings extracted and saved to {save_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./data/Amazon_Dataset', help='Data directory.')
    parser.add_argument('--set_name', type=str,
                        default='train', help='Dataset split name.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--embed_size', type=int, default=100,
                        help='Embedding dimension (hidden_channels).')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for ranking loss.')
    parser.add_argument('--p_norm', type=float, default=1.0,
                        help='Norm for distance (p_norm).')
    parser.add_argument('--sparse', action='store_true',
                        help='If set, use sparse gradients for embeddings.')
    parser.add_argument('--model', type=str, default='transe',
                        help='KGE model to use: transe, complex, distmult, rotate.')
    parser.add_argument('--log_dir', type=str, default='./tmp',
                        help='Directory to save checkpoints and embeddings.')
    parser.add_argument('--log_steps', type=int, default=100,
                        help='Logging interval (in steps).')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
    extract_embeddings(args)


if __name__ == '__main__':
    main()
