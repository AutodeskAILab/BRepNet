import argparse
from pytorch_lightning import Trainer
from pathlib import Path

from models.brepnet import BRepNet

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser = BRepNet.add_model_specific_args(parser)
opts = parser.parse_args()


def do_training():

    brepnet = BRepNet(opts)      

    print("Starting training")
    trainer = Trainer.from_argparse_args(opts)
    trainer.fit(brepnet)

    print("End training")


if __name__ == '__main__':
    do_training()