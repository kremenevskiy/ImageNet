import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


def name_from_path(path):
    return path.split('/')[-1][:-5]


def save_descriptors(names, descriptors):
    saved_paths = []
    for name, descriptor in zip(names, descriptors):
        descriptor = descriptor.reshape(1, -1)
        path = 'out/' + name
        np.savez(path, descriptor)
        saved_paths.append(path + '.npz')
    return saved_paths


def get_model(weights_path=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # FineTune ResNet18
    model = models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc.out_features = 200

    # Get pretrained weights
    if weights_path is None:
        weights_path = 'models/224/model_10_epoch.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    return model

def get_mapped_predictions(mapping_dict, predictions):
    mapped = [mapping_dict[pred] for pred in predictions]
    return mapped


class ImageFolderPath(datasets.ImageFolder):
    def __getitem__(self, index):
        original = super(ImageFolderPath, self).__getitem__(index)
        path = self.imgs[index][0]
        return original + (path,)


def inference(images_path, csv_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])

    ])

    image_dataset = ImageFolderPath(images_path, data_transforms)
    dataloader = data.DataLoader(image_dataset, batch_size=20, shuffle=False, num_workers=0)

    predictions = []
    descriptors_all = []
    paths_all = []

    # Get mapping from train full dataset
    class_mapping = ImageFolderPath('tiny-imagenet-200/train').class_to_idx
    class_mapping = {v:k for k, v in class_mapping.items()}


    for i, (inputs, labels, paths) in enumerate(dataloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs[:, :200]
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().flatten().tolist()
            predictions += get_mapped_predictions(class_mapping, preds)

            paths_all += np.array(paths).flatten().tolist()
            img_names = [name_from_path(path) for path in paths]

            # Get 1 * 64 descriptor
            descriptors = nn.AvgPool1d(10, 3)(outputs)
            descriptors = descriptors.numpy()
            descriptors_all += save_descriptors(img_names, descriptors)
        print(f'Working: {i+1} / {len(list(iter(dataloader)))}')
        sys.stdout.flush()

    # Save to DataFrame
    results = pd.DataFrame({'image_path': paths_all, 'predicted': predictions, 'descriptors': descriptors_all})
    results.to_csv(csv_path)

    return results


if __name__ == '__main__':
    argv = sys.argv
    inference(argv[1], argv[2])
