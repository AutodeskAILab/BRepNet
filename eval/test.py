import argparse
from pytorch_lightning import Trainer
from pathlib import Path

from models.brepnet import BRepNet

def do_testing(opts):

    trainer = Trainer.from_argparse_args(opts)
    brepnet = BRepNet.load_from_checkpoint(opts.model, opts=opts)

    print("Starting testing")
    trainer.test(brepnet)
    print("End testing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BRepNet.add_model_specific_args(parser)
    parser.add_argument("--model", type=str, required=True,  help="Model to load use for testing")
    opts = parser.parse_args()

    do_testing(opts)