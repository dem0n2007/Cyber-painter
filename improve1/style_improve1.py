import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 512
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def unnormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(t.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(t.device)
    return t * std + mean

content_img = transform(Image.open("stream_flower.jpeg")).unsqueeze(0).to(device)
style_img   = transform(Image.open("water lily2.jpg")).unsqueeze(0).to(device)

vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

content_layer = 'conv4_2'
style_layers  = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layer_map = {'0':'conv1_1','5':'conv2_1','10':'conv3_1',
             '19':'conv4_1','21':'conv4_2','28':'conv5_1'}

def get_features(img, model):
    feats = {}
    x = img
    for name, layer in model.named_children():
        x = layer(x)
        if name in layer_map:
            feats[layer_map[name]] = x
    return feats

def gram_matrix(t):
    b, c, h, w = t.size()
    f = t.view(b, c, h*w)
    return torch.bmm(f, f.transpose(1,2)) / (c*h*w)

def tv_loss(img):
    return (torch.sum(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:])) +
            torch.sum(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1])))

# 预计算目标（只算一次）
content_feats = get_features(content_img, vgg)[content_layer]
style_feats   = get_features(style_img, vgg)
style_grams   = {l: gram_matrix(style_feats[l]) for l in style_layers}

alpha = 1.0
gamma = 1e-6

# ========== 把一次风格迁移封装成函数 ==========
def run_style_transfer(beta, name, max_iter=300):
    gen_img = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([gen_img], max_iter=20)

    for step in range(max_iter):
        def closure():
            optimizer.zero_grad()
            feats = get_features(gen_img, vgg)

            c_loss = torch.mean((feats[content_layer] - content_feats) ** 2)

            s_loss = 0.0
            for l in style_layers:
                g = gram_matrix(feats[l])
                s_loss += torch.mean((g - style_grams[l]) ** 2)
            s_loss /= len(style_layers)

            t_loss = tv_loss(gen_img)
            total = alpha * c_loss + beta * s_loss + gamma * t_loss
            total.backward()
            with torch.no_grad():
                gen_img.clamp_(-2.0, 2.5)
            return total
        optimizer.step(closure)

    # 打印最终损失
    feats = get_features(gen_img, vgg)
    c = torch.mean((feats[content_layer] - content_feats) ** 2).item()
    s = sum(torch.mean((gram_matrix(feats[l]) - style_grams[l])**2).item()
            for l in style_layers) / len(style_layers)
    print(f"[{name}] beta={beta:.0e}  内容损失={c:.2f}  风格损失={s:.2f}")

    final = unnormalize(gen_img.squeeze(0)).clamp(0,1)
    transforms.ToPILImage()(final).save(f"output_{name}.jpg")

# ========== 三种风格强度 ==========
run_style_transfer(beta=1e4, name="weak")     # 弱风格
run_style_transfer(beta=1e5, name="medium")   # 中等风格
run_style_transfer(beta=1e7, name="strong")   # 强风格