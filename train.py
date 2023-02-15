import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_data, create_dataloader
from model import create_model
from trainer import Trainer


def train(root_dir, epochs, batch_size, logdir):
    # data
    train_faces, test_faces = load_data(root_dir)

    train_dl = create_dataloader(root_dir, train_faces, batch_size, training=True)
    test_dl = create_dataloader(root_dir, test_faces, batch_size, training=False)

    # model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    embedder = create_model(embedding_dims=256).to(device)

    # loss and optimizer
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    optimizer = torch.optim.Adam(embedder.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs - 10, epochs - 1], gamma=0.1)

    trainer = Trainer(loss_fn, optimizer, logdir)

    # training loop
    for epoch in range(epochs):
        # train
        loss_by_step, accuracy_by_step = [], []

        embedder.train()
        for anchor_img, pos_img, neg_img in tqdm(train_dl, desc=f'epoch {epoch}'):
            anchor_img, pos_img, neg_img = list(map(lambda x: x.to(device),
                                                    [anchor_img, pos_img, neg_img]))

            loss, accuracy = trainer.train_step(embedder, anchor_img, pos_img, neg_img)

            loss_by_step.append(loss)
            accuracy_by_step.append(accuracy)

        train_loss = sum(loss_by_step) / len(loss_by_step)
        train_accuracy = sum(accuracy_by_step) / len(accuracy_by_step)

        scheduler.step()

        torch.save({'epoch': epoch,
                    'model_state_dict': embedder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join(logdir, 'checkpoint.pt'))

        print(f'[train] | loss: {loss:.5f}, accuracy: {accuracy:.4f}')

        # test
        loss_by_step, accuracy_by_step = [], []

        embedder.eval()
        for anchor_img, pos_img, neg_img in test_dl:
            anchor_img, pos_img, neg_img = list(map(lambda x: x.to(device),
                                                    [anchor_img, pos_img, neg_img]))

            loss, accuracy = trainer.eval_step(embedder, anchor_img, pos_img, neg_img)

            loss_by_step.append(loss)
            accuracy_by_step.append(accuracy)

        test_loss = sum(loss_by_step) / len(loss_by_step)
        test_accuracy = sum(accuracy_by_step) / len(accuracy_by_step)

        print(f'[test] | loss: {loss:.5f}, accuracy: {accuracy:.4f}')

        trainer.writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)
        trainer.writer.add_scalars('accuracy', {'train': train_accuracy, 'test': test_accuracy}, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='extracted faces path')
    parser.add_argument('--epochs', type=int, help='training epochs')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--logdir', default='.logs/', help='path to save logs')

    args = parser.parse_args()

    train(args.root_dir, args.epochs, args.batch, args.logdir)
