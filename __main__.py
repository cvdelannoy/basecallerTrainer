import argparse
import sys

import argparse_dicts
import run_ordinal_brnn

def main(args=None):
    if args == None:
        args = sys.argv[1:]

    commands = [
        ('train an rnn and output trained parameters',[
            ('train_ordinal_brnn',
             'Train an ordinal bidirectional rnn',
             argparse_dicts.get_brnn_parser(),

             resquiggle.resquiggle_main)]),


if __name__ == '__main__':
    main()