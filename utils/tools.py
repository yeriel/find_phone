import argparse
from operator import mul
from functools import reduce


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Train a phone finder model.')
    parser.add_argument('data_folder', type=str,
                        help='Path to the folder with labeled images and labels.txt')
    return parser.parse_args()


def count_module_parameters(module):
    """Retorna la cantidad de parámetros del módulo."""
    return sum(reduce(mul, parameter.size()) for parameter in module.parameters())
