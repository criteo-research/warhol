import numpy as np
import torch
import torchvision
from torchvision import models, transforms
import clip
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor
from torch.nn import Parameter
from omegaconf import OmegaConf
from PIL import Image
import time
from torch.utils.data import DataLoader

from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, DALLE, VQGanVAE, WARHOL


def load_model(model_path, vqgan_model_path, vqgan_config_path, warhol=True):
    timet = time.time()
    assert model_path.exists(), 'trained model must exist'
    load_obj = torch.load(str(model_path))
    model_params, vae_params, weights = \
        load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')
    model_params.pop('vae', None)

    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    else:
        vae = VQGanVAE(vqgan_model_path, vqgan_config_path)
    if warhol:
        model = WARHOL(vae = vae, **model_params).cuda()
    else:
        model = DALLE(vae = vae, **model_params).cuda()
    model.load_state_dict(weights)
    model_size = vae.image_size
    print('Model loaded', time.time()-timet)
    return model


def create_warhol_loader(folder, text_len, image_size, tokenizer, batch_size=16):
    ds = TextImageDataset(
        folder,
        text_len=text_len,
        image_size=image_size,
        truncate_captions=True,
        tokenizer=tokenizer
    )
    dataloader = DataLoader(ds, batch_size=batch_size, drop_last=True)
    return dataloader

def show(img):
    npimg = img.cpu().numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def stack_reconstructions(images, text, num_rows, num_columns):
    #assert input.size == x1.size == x2.size == x3.size
    w, h = images[0].shape[1], images[0].shape[2]
    img = Image.new("RGB", (num_columns*w, num_rows*h))
    for i in range(num_rows):
        for j in range(num_columns):
            im = images[i*num_columns+j] 
            #im = preprocess_vqgan(im) if j==0 else im                
            im = custom_to_pil(im)
            img.paste(im, (j*w, i*h))
    #ImageDraw.Draw(img).text(((i%5)*w, int(i/5)), f'{title}', (255, 255, 255), font=font)
    img.save(text+".png")
    return img

def plot_images_reconstruction(images, text, num_rows, num_columns):
    torchvision.utils.save_image(images, text+'.jpg', nrow=num_columns)
    stack_reconstructions(images, text, num_rows, num_columns)
    #plt.clf()
    #fig=plt.figure(figsize=(10, 90), dpi=384)
    #for i in range(len(images)):
    #    ax_i = fig.add_subplot(num_rows, num_columns, i+1)
    #    show(images[i])
    #text = "_".join(text.split())
    #plt.savefig(text[:50], bbox_inches="tight")
           
def plot_images(images, text, num_rows, num_columns):
    plt.clf()
    fig=plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        ax_i = fig.add_subplot(num_rows, num_columns, i+1)
        show(images[i])
    text = "_".join(text.split())
    plt.savefig(text[:50], bbox_inches="tight")
    
def get_additional_layers(num_features):
    hidden_layer = 32
    if hidden_layer:
        return nn.Sequential(nn.Linear(num_features, hidden_layer),
                             nn.ReLU(),
                             nn.Linear(hidden_layer, 1))
    return nn.Linear(num_features, 1)

def get_model_pclick(path):
    model = models.resnet50(pretrained=True)
    model.fc = get_additional_layers(model.fc.in_features)
    model.to('cuda')
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def get_additional_layers(num_features):
    return nn.Sequential(nn.Dropout(p=0.2),
                    nn.Linear(num_features, 1))

def get_model_efficientnet2(filename):
    m = timm.create_model('tf_efficientnetv2_s_in21k', pretrained=True, num_classes=0)
    o = m(torch.randn(2, 3, 224, 224))
    model = nn.Sequential(m, get_additional_layers(o.shape[1]))
    return model

def get_pclick_forward(model, transform):
    model.eval()
    def pclik(image):
        pil_image = transforms.ToPILImage()(image)
        return model(transform(pil_image))
    return pclik


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming_transformers.taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming_transformers.taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming_transformers.taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def reconstruct_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    #x = preprocess_vqgan(x)
    z, _, [_, _, indices] = model.encode(x)
    #print(f"VQGAN: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec