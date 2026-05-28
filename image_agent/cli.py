import argparse
from .train import train
from .infer import predict
from .reverse_search import build_index, search_similar_images, results_to_csv


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

    b = sub.add_parser('build-index')
    b.add_argument('--data', required=True)
    b.add_argument('--index', default='image_agent/reverse_index.npz')
    b.add_argument('--limit', type=int, default=None)

    s = sub.add_parser('search')
    s.add_argument('--query', required=True)
    s.add_argument('--index', default='image_agent/reverse_index.npz')
    s.add_argument('--top-k', type=int, default=5)
    s.add_argument('--out', default='image_agent/reverse_search_results.csv')

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args.data, epochs=args.epochs, batch_size=args.batch, out_path=args.out)
    elif args.cmd == 'infer':
        print(predict(args.model, args.image))
    elif args.cmd == 'build-index':
        out = build_index(args.data, args.index, args.limit)
        print(f'Index saved to {out}')
    elif args.cmd == 'search':
        results = search_similar_images(args.query, args.index, args.top_k)
        results_to_csv(results, args.out)
        for row in results:
            print(row)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
