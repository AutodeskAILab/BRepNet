import argparse
from pytorch_lightning import Trainer
from pathlib import Path

from models.brepnet import BRepNet

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser = BRepNet.add_model_specific_args(parser)
opts = parser.parse_args()


def do_testing():

    trainer = Trainer.from_argparse_args(opts)
    brepnet = BRepNet.load_from_checkpoint(opts.test_with_pretrained_model, opts=opts)

    print("Starting testing")
    trainer.test(brepnet)
    print("End testing")


if __name__ == '__main__':
    do_testing()