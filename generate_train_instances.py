# Use the generator from the td_opswtw competition to generate a dataset of new instances.
# In particular, creates instances.npy which contains a numpy array of shape [# instances, # nodes per instance, 7]
# which specifies properties for each node of each instance and adjs.npy of shape [# instances, # nodes, # nodes]

from generator.op.instances import InstanceGenerator
import argparse
import os
import numpy as np


def print_and_log(string, log_file):
    print(string)
    if log_file is not None and os.path.isfile(log_file):
        with open(log_file, 'a') as f:
            f.write(string + '\n')


def save_data(instances_filepath, adjs_filepath, data, adjs, old_data, old_adjs, log_file):
    if old_data is not None:
        data = np.concatenate([old_data, data])
        adjs = np.concatenate([old_adjs, adjs])

    print_and_log(f'Saving {data.shape[0]} instances to file.', log_file)
    np.save(instances_filepath, data)
    np.save(adjs_filepath, adjs)


def generate_instances(args):
    # create log file
    if args.log_file is not None:
        f = open(args.log_file, 'w')
        f.close()

    print_and_log('Generating Train instances.', args.log_file)
    for k, v in args.__dict__.items():
        print_and_log(f'{k.upper()} = {v}\n', args.log_file)

    if not os.path.isdir(args.save_dir):
        print_and_log(
            f'{args.save_dir} Directory not found relative to cwd: {os.getcwd()}', args.log_file)
        exit(1)

    # Filepath to save instance data numpy object
    instances_filepath = os.path.join(args.save_dir, 'instances.npy')
    # Filepath to save adjacency matrix numpy object
    adjs_filepath = os.path.join(args.save_dir, 'adjs.npy')

    old_data = None
    old_adjs = None
    n_existing_instances = 0

    # retrieve existing instances
    if os.path.isfile(instances_filepath) and os.path.isfile(adjs_filepath):
        old_data = np.load(instances_filepath)
        old_adjs = np.load(adjs_filepath)
        n_existing_instances = old_data.shape[0]
        print_and_log(
            f'Found {n_existing_instances} instances.', args.log_file)

    data = []
    adjs = []
    if n_existing_instances >= args.n_instances:
        print_and_log(
            f'{n_existing_instances} already exist. No new instances will be generated', args.log_file)
        exit(1)

    print_and_log(
        f'Generating {args.n_instances - n_existing_instances} new instances.', args.log_file)

    for i in range(args.n_instances-n_existing_instances):
        gen = InstanceGenerator(n_instances=1, n_nodes=args.problem_s)
        x, adj = gen.generate_instance_files(save=False)
        data.append(x.to_numpy())
        adjs.append(adj.to_numpy())

        if (i+1) % args.save_freq == 0:
            save_data(instances_filepath,
                      adjs_filepath,
                      np.array(data).astype(np.float32),
                      np.array(adjs).astype(np.float32),
                      old_data,
                      old_adjs,
                      args.log_file)

    save_data(instances_filepath,
              adjs_filepath,
              np.array(data).astype(np.float32),
              np.array(adjs).astype(np.float32),
              old_data,
              old_adjs,
              args.log_file)


if __name__ == '__main__':
    # python generate_train_instances.py train_data/20 20 --n_instances 5000 --save_freq 100
    # Generates train instances in train_data/20 folder, with 20 nodes per instance. Creates 5000 instances, 
    # saving every 100 instances.
    parser = argparse.ArgumentParser(description='Generate Instances')
    parser.add_argument(
        'save_dir', help="Location to save generated instances")
    parser.add_argument('problem_s', type=int,
                        help="Problem size/number of nodes per instance")
    parser.add_argument('--n_instances', type=int,
                        help="Target total number of instances in dataset. Defaults to 100k", default=100000)
    parser.add_argument("--save_freq", type=int,
                        help="Number of instances between saving output. Defaults to 10k", default=10000)
    parser.add_argument(
        "--log_file", help="File to write logs to. Defaults to None", default=None)
    args = parser.parse_args()
    generate_instances(args)
