import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default='../garage/pems08/2/')
parser.add_argument('--gpu', type=int, default=0, help='the number of the gpu to use')
parser.add_argument('--devices', type=str, default='0,1', help='the number of the gpu to use')
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--seed', type=int, default=20, help='seed')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--lr_decay_steps', type=int, nargs='+',default=[50,105,145], help='lr_decay_steps')
parser.add_argument('--lr_decay_rate', type=float, default=0.10, help='lr_decay_rate')
parser.add_argument('--filename', type=str, default='pems04')
parser.add_argument('--train-ratio', type=float, default=0.6, help='the ratio of training dataset')
parser.add_argument('--valid-ratio', type=float, default=0.2, help='the ratio of validating dataset')
parser.add_argument('--his-length', type=int, default=12, help='the length of history time series of input')
parser.add_argument('--pred-length', type=int, default=12, help='the length of target time series for prediction')

parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
parser.add_argument('--thres2', type=float, default=6, help='the threshold for the spatial matrix')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')

parser.add_argument('--log', action='store_true', help='if write log to files')

parser.add_argument('--embed_dim', default=10, type=int)
parser.add_argument('--num_nodes', default=307, type=int)

args = parser.parse_args()


args.devices = args.devices.replace(' ','')
device_ids = args.devices.split(',')

args.device_ids = [int(id_) for id_ in device_ids]
args.gpu = args.device_ids[0]
