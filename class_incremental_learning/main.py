import json
import argparse
from trainer import train

def main():
    parser = setup_parser()
    args = parser.parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    cli_seed = args.pop("seed", None)  # Save CLI seed before config overwrites it
    args.update(param)  # Add parameters from json
    if cli_seed is not None:
        args["seed"] = cli_seed  # CLI seed overrides config

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--seed', type=int, nargs='+', default=None,
                        help='Random seed(s), e.g. --seed 42 or --seed 42 123. Overrides config value.')
    return parser

if __name__ == '__main__':
    main()
