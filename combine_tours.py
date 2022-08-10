import json
import os
import argparse


def run_combine(args):
    instances = sorted(os.listdir(args.tour_dir))
    out_dict = dict()
    for inst in instances:
        if inst.endswith('.json') and inst.startswith('instance'):
            f = open(os.path.join(args.tour_dir, inst))
            out_dict.update(json.load(f))
            f.close()

    save_path = os.path.join(args.tour_dir, 'submission.json')
    with open(save_path, 'w') as f:
        json.dump(out_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine Instance Tours')
    parser.add_argument('tour_dir', help='Directory where Instance Tours are saved')

    run_args = parser.parse_args()
    run_combine(run_args)
