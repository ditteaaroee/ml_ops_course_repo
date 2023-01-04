import argparse
import sys
import torch
import click
from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = mnist(train=True)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epoch = 5
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
    torch.save(model.state_dict(), 'trained_model.pt')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist(train=False)
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
    
    correct, total = 0, 0
    for batch in dataloader:
        x, y = batch
        
        preds = model(x)
        preds = preds.argmax(dim=-1)
        
        correct += (preds == y).sum().item()
        total += y.numel()
        
    print(f"Test set accuracy {correct/total}")



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()



    
    
    
    