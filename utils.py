import os

def parser_bool(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})


def cifar_config(args):
    if args.dataset == 'cifar100':
        if args.idn_noise == 0.6:
            args.p_threshold = 0.3
            args.d_threshold = 0.3
        elif args.idn_noise == 0.4:
            args.p_threshold = 0.5
            args.d_threshold = 0
        elif args.idn_noise == 0.2:
            args.p_threshold = 0.4
            args.d_threshold = 0
    elif args.dataset == 'cifar10':
        args.p_threshold = 0.4
        args.d_threshold = 0
