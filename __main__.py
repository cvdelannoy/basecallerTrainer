#!/usr/bin/python3
import argparse
import sys

import argparse_dicts
import train_ordinal_brnn

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    commands = [
        ('train_ordinal_brnn',
        'Train an ordinal bidirectional rnn',
        argparse_dicts.get_brnn_parser(),
        train_ordinal_brnn.main)
        ]

    parser = argparse.ArgumentParser(
        prog='basecallerTrainer',
        description='A flexible tool for training and subsequently recognizing patterns in MinION data.'
    )
    subparsers = parser.add_subparsers(
        title='commands'
    )

    for cmd, hlp, ap, fnc in commands:
            subparser = subparsers.add_parser(cmd, add_help=False, parents=[ap,])
            subparser.set_defaults(func=fnc)
    args = parser.parse_args(args)
    args.func(args)

if __name__ == '__main__':
    main()