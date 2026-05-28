import argparse
from .train import train
from .infer import predict


def main():
    parser = argparse.ArgumentParser(description='Image Agent CLI')
    sub = parser.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--epochs', type=int, default=3)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--out', default='model.pth')

    i = sub.add_parser('infer')
    i.add_argument('--model', required=True)
    i.add_argument('--image', required=True)

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args.data, epochs=args.epochs, batch_size=args.batch, out_path=args.out)
    elif args.cmd == 'infer':
        print(predict(args.model, args.image))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
