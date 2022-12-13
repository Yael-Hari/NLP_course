import torch

from preprocessing import SentencesEmbeddingDataset

ds = SentencesEmbeddingDataset(
    vec_dim=500,
    list_embedding_paths=["glove-twitter-200", "word2vec-google-news-300"],
    list_vec_dims=[200, 300],
)
batch_size = 32
train_dataloader, dev_dataloader, test_dataloader = ds.get_data_loaders(
    batch_size=batch_size
)

ds.embedding_model = None

torch.save(
    (train_dataloader, dev_dataloader, test_dataloader), f"concated_ds_{batch_size}_withO.pkl"
)

# this is how to get the loaders:
# train_dataloader, dev_dataloader, test_dataloader = torch.load(f'concated_ds_{batch_size}.pkl')
