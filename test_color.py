import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=12, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/exp1", help='path of log files')
parser.add_argument("--test_data", type=str, default='video', help='test on Set12 or Set68 or Setcolor')
parser.add_argument("--test_noiseL", type=float, default=5, help='noise level used on test set')
parser.add_argument("--save_dir", type=str, default="result", help='save_dir')
opt = parser.parse_args()

def normalize(data):
    return data/255.0
def show(x, title=None, cbar=False, figsize=None):
    
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
    
    
def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    print(net)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'netB50color.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    #gray images: png
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.MP4'))
    files_source.sort()
    # process data
    psnr_test = 0
    
    
    
    cap = cv2.VideoCapture(files_source[0])
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter('outpy.avi',fourcc, 60, (2*1280,720))
    curFrame=0
    while (cap.isOpened):
        curFrame=curFrame+1
      
        ret,frame=cap.read()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        if ret==True:
            Img=frame
            color_flag = 1
            
            if color_flag==0:
                # imag  +0e
                Img = normalize(np.float32(Img[:,:,0]))
                Img = np.expand_dims(Img, 0)
                Img = np.expand_dims(Img, 1)
                ISource = torch.Tensor(Img)
                # noise
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
                # noisy image
                INoisy = ISource + noise
                ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
                with torch.no_grad(): # this can save much memory
                    Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
                ## if you are using older version of PyTorch, torch.no_grad() may not be supported
                # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
                # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
                psnr = batch_PSNR(Out, ISource, 1.)
                psnr_test += psnr
                print("%s PSNR %f" % (f, psnr))
            
            
        
        #For color images
            else:
                dims = [0,1,2]
                times=0
                
#                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                Img2 = Img
              #  print(Img2.shape)
                s1Img = np.zeros(Img.shape,dtype=np.float32)
                s2Img = np.zeros(Img.shape,dtype=np.float32)
                noiseAll =  np.zeros(Img.shape,dtype=np.float32)
               
                resultImg=np.zeros(Img.shape,dtype=np.float32)
                # imag  +0e
                for i in dims:
                    Img = normalize(np.float32(Img2[:,:,i]))
                    
                    s1Img[:,:,i] = Img
                
                
                    
                
                    
                s1Img = s1Img.transpose((2,0,1))
                s1Img = np.expand_dims(s1Img,0)
                ISource = torch.Tensor(s1Img)
                
               # print(ISource.shape)
                # noisy image
                INoisy = ISource 
                ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
                with torch.no_grad(): # this can save much memory
                    noise_get=model(INoisy)
                    Out = torch.clamp(INoisy-noise_get, 0., 1.)
                ## if you are using older version of PyTorch, torch.no_grad() may not be supported
                # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
                # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
#                psnr = batch_PSNR(Out, ISource, 1.)
#                psnr_test += psnr
#                print("%s PSNR %f" % (f, psnr))
                resultImg=Out.cpu().numpy()
                sImg=INoisy.cpu().numpy()
                noiseAll=noise_get.cpu().numpy()
                sImg = np.squeeze(sImg)
                resultImg = np.squeeze(resultImg)
                sImg = sImg.transpose((1,2,0))
                resultImg = resultImg.transpose((1,2,0))
                a=np.hstack((sImg,resultImg))
                
                a=np.uint8(a*255)
                save_result(a, path=os.path.join(opt.save_dir, str(times)+'.png'))
                out.write(a)
                times=times+1
                print('---------curFrame: %d' %(curFrame))
    
    cap.release()
    out.release()
                
        
           
            
            
#    psnr_test /= len(files_source)*3
#    print("\nPSNR on test data %f" % psnr_test)
    

if __name__ == "__main__":
    main()
