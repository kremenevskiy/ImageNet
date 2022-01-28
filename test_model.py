import torch
import os, sys, time


def test_model(model, data_loaders, dataset_sizes, criterion, optimizer, phases=['val']):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    time_start = time.time()
    for phase in phases:

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(data_loaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                outputs = outputs[:, :200]
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(data_loaders[phase]), loss.item() * inputs.size(0)),
                  end="")
            # sys.stdout.flush()

            
            print(f'Running corrects: {torch.sum(preds == labels.data)}/{100}')
            print()
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

    print()
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print('Total corrects: {}/{}'.format(running_corrects, dataset_sizes[phase]))
    print()
    print('-'*10)

    time_elapsed = time.time() - time_start
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
