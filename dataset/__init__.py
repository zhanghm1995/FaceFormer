import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_3dmm_dataset(config, split, shuffle=None):
    """Get the dataset contains 2D image and 3D information

    Args:
        config (dict): config parameters
        split (str): train or val
        shuffle (bool, optional): Whether shuffle. Defaults to None.

    Returns:
        DataLoader: the torch dataloader
    """
    from .face_3dmm_dataset import Face3DMMDataset

    dataset = Face3DMMDataset(data_root=config['data_root'], 
                              split=split, 
                              fetch_length=config['fetch_length'])

    data = dataset[36]
    data = dataset[37]
    
    
    ## minibatch for debuging
    # sub_dataset = []
    # for idx in range(8):
    #     sub_dataset.append(data)
    # for idx in range(8):
    #     sub_dataset.append(dataset[36])

    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(split=="train") if shuffle is None else shuffle,
        num_workers=config['number_workers'],
        # pin_memory=True,
        pin_memory=False,
        collate_fn=collate_fn
    )
    return data_loader