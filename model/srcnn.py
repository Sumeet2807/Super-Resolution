import torch.nn as nn

import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
from torchvision.utils import save_image, make_grid
from ts.torch_handler.base_handler import BaseHandler
from torch.autograd import Variable
import torch.nn as nn
from .utils import calc_psnr
from PIL import Image
import requests
from io import BytesIO

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


    def train_model(self,train_loader,validation_loader,config):

        cuda = torch.cuda.is_available()

        hr_shape = (config.hr_height, config.hr_width)

        bicubic_upscaler = transforms.Resize((config.hr_width, config.hr_height), Image.BICUBIC)
        # Losses
        criterion_loss = torch.nn.MSELoss()

        if cuda:
            self = self.cuda()
            criterion_loss = criterion_loss.cuda()

        if config.epoch != 0:
            # Load pretrained models
            self.load_state_dict(torch.load(config.model_checkpoint_dir + "/srcnn_%d.pth"))

        # Optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, betas=(config.b1, config.b2))
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        self.train_psnr_log = []
        self.train_loss_log = []
        self.valid_psnr_log = []
        self.valid_loss_log = []

        # ----------
        #  Training
        # ----------

        for epoch in range(config.epoch, config.n_epochs):
            avg_loss = 0
            avg_psnr = 0
            for i, imgs in enumerate(train_loader):

                # Configure model input
                imgs_lr = Variable(bicubic_upscaler(imgs["lr"].type(Tensor)))
                imgs_hr = Variable(imgs["hr"].type(Tensor))


                # ------------------
                #  Train 
                # ------------------

                optimizer.zero_grad()

                imgs_sr = self(imgs_lr)
                loss = criterion_loss(imgs_sr,imgs_hr)
                loss.backward()
                optimizer.step()
                # --------------
                #  Log Progress
                # --------------
                avg_loss = ((avg_loss*i) + loss.item())/(i+1)
                avg_psnr = ((avg_psnr*i) + calc_psnr(imgs_sr,imgs_hr).item())/(i+1)

                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [Loss: %f]\n"
                    % (epoch, config.n_epochs, i, len(train_loader), loss.item())
                )

                batches_done = epoch * len(train_loader) + i
                if batches_done % config.sample_interval == 0:
                    self.train_loss_log.append(avg_loss)
                    self.train_psnr_log.append(avg_psnr)            
                    avg_loss = 0
                    avg_psnr = 0
                    for j, imgs in enumerate(validation_loader):

                        # Configure model input
                        imgs_lr = Variable(bicubic_upscaler(imgs["lr"].type(Tensor)))
                        imgs_hr = Variable(imgs["hr"].type(Tensor))
                        imgs_sr = self(imgs_lr)

                        avg_loss = ((avg_loss*j) + criterion_loss(imgs_sr,imgs_hr).item())/(j+1)
                        avg_psnr = ((avg_psnr*j) + calc_psnr(imgs_sr,imgs_hr).item())/(j+1)

                        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                        imgs_sr = make_grid(imgs_sr, nrow=1, normalize=True)
                        img_grid = torch.cat((imgs_lr, imgs_sr, imgs_hr), -1)
                        save_image(img_grid, config.results + "/validation_%d.png" % batches_done, normalize=False)
                    
                    self.valid_loss_log.append(avg_loss)
                    self.valid_psnr_log.append(avg_psnr)
                    

            if config.checkpoint_interval != -1 and epoch % config.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.state_dict(), config.model_checkpoint_dir + "srcnn_%d.pth" % epoch)







class serveHandler(BaseHandler):

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        image_url = data[0]['body']['url']

        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        print(np.array(img).shape)
        # print(data.shape)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        data = torch.ones((16,64))
        return torch.as_tensor(data, device=self.device)

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """
        
        data = torch.argmax(data,axis=1)
        print(data.tolist())
        return [data.tolist()]

