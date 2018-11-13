import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from flask import send_file,render_template

import data_loader_evaluate
from torch.autograd import Variable
from model import _G_xvz, _G_vzx
import pdb
from PIL import Image
import flask
import io
import os
# base_path = os.path.dirname(__file__)
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torchvision import transforms as T
# app = flask.Flask(__name__,static_folder=os.path.join(base_path,'static'),template_folder=os.path.join(base_path,'templates'))

# dd = pdb.set_trace

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test_dir", type=str, default="test_data")
parser.add_argument("-d", "--data_list", type=str, default="list_test.txt")
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument('--outf', default='evaluate', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='pretrained_model', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

args = parser.parse_args()
print(args)

# try:
#     os.makedirs(os.path.join(base_path,args.outf))
# except OSError:
#     pass

# print(torch.cuda.is_available())
# if torch.cuda.is_available() and not args.cuda:
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# need initialize!!
# G_xvz = _G_xvz()
# G_vzx = _G_vzx()
def init():
    base_path = os.path.dirname(__file__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dd = pdb.set_trace
    try:
        os.makedirs(os.path.join(base_path, args.outf))
    except OSError:
        pass
    print(torch.cuda.is_available())
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    G_xvz = _G_xvz()
    G_vzx = _G_vzx()
    train_list = os.path.join(base_path,args.data_list)
    train_loader = torch.utils.data.DataLoader(
        data_loader_evaluate.ImageList( train_list, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    x = torch.FloatTensor(args.batch_size, 3, 128, 128)
    x_bar_bar_out = torch.FloatTensor(10, 3, 128, 128)

    v_siz = 9
    z_siz = 128 - v_siz
    v = torch.FloatTensor(args.batch_size, v_siz)
    z = torch.FloatTensor(args.batch_size, z_siz)

    if args.cuda:
        G_xvz = torch.nn.DataParallel(G_xvz).cuda()
        G_vzx = torch.nn.DataParallel(G_vzx).cuda()

        x = x.cuda()
        x_bar_bar_out = x_bar_bar_out.cuda()
        v = v.cuda()
        z = z.cuda()

    x = Variable(x)
    x_bar_bar_out = Variable(x_bar_bar_out)
    v = Variable(v)
    z = Variable(z)
    load_pretrained_model(G_xvz, os.path.join(base_path, args.modelf), 'netG_xvz.pth')
    load_pretrained_model(G_vzx, os.path.join(base_path, args.modelf), 'netG_vzx.pth')

    batch_size = args.batch_size
    cudnn.benchmark = True
    G_xvz.eval()
    G_vzx.eval()
    return x,v,z,x_bar_bar_out,G_xvz,G_vzx,batch_size,train_loader


def L1_loss(x, y):
    return torch.mean(torch.sum(torch.abs(x-y), 1))


# x = torch.FloatTensor(args.batch_size, 3, 128, 128)
# x_bar_bar_out = torch.FloatTensor(10, 3, 128, 128)
#
# v_siz = 9
# z_siz = 128-v_siz
# v = torch.FloatTensor(args.batch_size, v_siz)
# z = torch.FloatTensor(args.batch_size, z_siz)
#
# if args.cuda:
#     G_xvz = torch.nn.DataParallel(G_xvz).cuda()
#     G_vzx = torch.nn.DataParallel(G_vzx).cuda()
#
#     x = x.cuda()
#     x_bar_bar_out = x_bar_bar_out.cuda()
#     v = v.cuda()
#     z = z.cuda()
#
# x = Variable(x)
# x_bar_bar_out = Variable(x_bar_bar_out)
# v = Variable(v)
# z = Variable(z)

def load_pretrained_model(net, path, name):
    state_dict = torch.load('%s/%s' % (path,name),map_location='cpu')
    own_state = net.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)
        print('load weights %s' % name)

# load_pretrained_model(G_xvz, os.path.join(base_path,args.modelf), 'netG_xvz.pth')
# load_pretrained_model(G_vzx, os.path.join(base_path,args.modelf), 'netG_vzx.pth')
#
# batch_size = args.batch_size
# cudnn.benchmark = True
# G_xvz.eval()
# G_vzx.eval()

# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route("/predict", methods=["POST"])
def evaluate():
    x, v, z, x_bar_bar_out, G_xvz, G_vzx, batch_size,train_loader=init()
    # if flask.request.method == "POST":
    #     test_data = flask.request.files
    #     image = flask.request.files["image"].read()
    #     # image = io.BytesIO(image)
    #     image = Image.open(io.BytesIO(image))
    #     img = T.ToTensor()(image)
    #     img_norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    #     img = img_norm.view([1,3,128,128])
    for i, (data) in enumerate(train_loader):
    # for i,(data) in enumerate(image):
    #     print(i)
        img = data
    x.data.resize_(img.size())
    x.data.copy_(img)

    x_bar_bar_out.data.zero_()
    v_bar, z_bar = G_xvz(x)

    for one_view in range(9):
        v.data.zero_()
        for d in range(img.size(0)):
            v.data[d][one_view] = 1
        exec('x_bar_bar_%d = G_vzx(v, z_bar)' % (one_view))
    data={}
    for d in range(batch_size):
        x_bar_bar_out.data[0] = x.data[d]
        for one_view in range(9):
            exec('x_bar_bar_out.data[1+one_view] = x_bar_bar_%d.data[d]' % (one_view))
        vutils.save_image(x_bar_bar_out.data[5],'%s/%d_x_bar_bar.png' % (os.path.join(os.path.dirname(__file__),args.outf), batch_size+d), nrow = 10, normalize=True, pad_value=255)
    #     byte_io = io.BytesIO()
    #     with open(os.path.join(os.path.dirname(__file__),args.outf)+'/1_x_bar_bar.png','rb')as f:
    #         byte_image = f.read()
    #     byte_io.write(byte_image)
    #     byte_io.seek(0)
    # return send_file(byte_io,mimetype="image/jpeg")

if __name__ == '__main__':
    main()