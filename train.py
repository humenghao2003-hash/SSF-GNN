import torch
import torch.nn as nn
from dataset import DatasetAnomaly, class_dict
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import SSFGNN
from torch.distributed.elastic.multiprocessing.errors import record


def reduce_stats(confusion, sum_loss, total_samples, device, distributed):
    if distributed:
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
        stats = torch.tensor([sum_loss, float(total_samples)], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        sum_loss = stats[0].item()
        total_samples = int(stats[1].item())
    return confusion, sum_loss, total_samples



@record
def main():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    # 1. load dataset
    train_path = "data/train"
    test_path = "data/test"
    num_class = 7
    binary = True if num_class == 2 else False
    batch_size = 16
    train_dataset = DatasetAnomaly(train_path, transform=True, load_in_memory=False, use_indices=True, binary_classify=binary)
    test_dataset = DatasetAnomaly(test_path, transform=False, load_in_memory=False, use_indices=True, binary_classify=binary)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    valid_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=0, pin_memory=False
    )
    valid_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=False, sampler=valid_sampler,
        num_workers=0, pin_memory=False
    )

    # 2. load model
    model = SSFGNN(num_class, 5).to(device)
    if distributed:
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    # 3. prepare super parameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. train
    val_acc_list = []
    out_dir = "checkpoints_new/"
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    if distributed:
        dist.barrier()
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        total_samples = 0
        confusion = torch.zeros(num_class, num_class, dtype=torch.int64, device=device)

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)  # [B, C, H, W] 或 [B, C]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss
            sum_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # GPU prediction
            predicted = outputs.argmax(1)

            # Confusion matrix (GPU)
            labels_flat = labels.view(-1)
            preds_flat = predicted.view(-1)
            inds = num_class * labels_flat + preds_flat
            cm = torch.bincount(inds, minlength=num_class ** 2).reshape(num_class, num_class)
            confusion += cm
        # Calculate evaluation metrics
        confusion, sum_loss, total_samples = reduce_stats(
            confusion, sum_loss, total_samples, device, distributed
        )
        confusion_np = confusion.cpu().numpy()
        iou = np.diag(confusion_np) / (confusion_np.sum(1) + confusion_np.sum(0) - np.diag(confusion_np))
        miou = np.nanmean(iou)
        oa = np.trace(confusion_np) / confusion_np.sum()
        avg_loss = sum_loss / total_samples
        if is_main:
            print('[epoch:%d, iter:%d] Loss: %.03f | mIoU: %.3f%% | OA: %.3f%%'
              % (epoch + 1, batch_idx, avg_loss, 100. * miou, 100. * oa))
        if is_main:
            print(class_dict)
        if is_main:
            print("IoU:", np.round(iou * 100, 3))

        # scheduler.step()


        # get the ac with testdataset in each epoch
        if is_main:
            print('Waiting Val...')
        model.eval()
        confusion = torch.zeros(num_class, num_class, dtype=torch.int64, device=device)
        sum_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_dataloader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                # loss = criterion(outputs, labels)
                loss = criterion(outputs, labels)
                sum_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # GPU prediction
                predicted = outputs.argmax(1)

                # Confusion matrix (GPU)
                labels_flat = labels.view(-1)
                preds_flat = predicted.view(-1)
                inds = num_class * labels_flat + preds_flat
                cm = torch.bincount(inds, minlength=num_class ** 2).reshape(num_class, num_class)
                confusion += cm

        # Calculate evaluation metrics
        confusion, sum_loss, total_samples = reduce_stats(
            confusion, sum_loss, total_samples, device, distributed
        )
        confusion_np = confusion.cpu().numpy()
        iou = np.diag(confusion_np) / (confusion_np.sum(1) + confusion_np.sum(0) - np.diag(confusion_np))
        miou = np.nanmean(iou)
        oa = np.trace(confusion_np) / confusion_np.sum()
        avg_loss = sum_loss / total_samples
        if is_main:
            print('[epoch:%d, iter:%d] Loss: %.03f | mIoU: %.3f%% | OA: %.3f%%'
              % (epoch + 1, batch_idx, avg_loss, 100. * miou, 100. * oa))
        if is_main:
            print(class_dict)
        if is_main:
            print("IoU:", np.round(iou * 100, 3))

        if is_main:
            val_acc_list.append(oa)

        if is_main and oa == max(val_acc_list):
            torch.save((model.module if distributed else model).state_dict(), out_dir + "best.pt")
            print("save epoch {} model".format(epoch + 1))

        if is_main:
            torch.save((model.module if distributed else model).state_dict(), out_dir + "last.pt")
        if distributed:
            dist.barrier()


    if is_main:
        print("Final highest accuracy: {}".format(max(val_acc_list)))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()