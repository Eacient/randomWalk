import torch
def transform_weight(path, new_path='new.pth'):
    dict = torch.load(path)['model']
    new_dict={}
    for k, v in dict.items():
        if 'gvae' in k:
            continue
        if 'backbone' in k:
            new_key = k[9:]
        else:
            new_key = k
        print(new_key)
        new_dict[new_key] = v
    torch.save(new_dict, new_path)

if __name__ == "__main__":
    transform_weight('/root/autodl-tmp/checkpoint-5.pth', 'prior_05.pth')