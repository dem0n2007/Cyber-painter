import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 512
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),   # 固定方形，避免尺寸不一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========== 反归一化 ==========
def unnormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(t.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(t.device)
    return t * std + mean

content_img = transform(Image.open("stream_flower.jpeg")).unsqueeze(0).to(device)
style_img_oil   = transform(Image.open("water lily2.jpg")).unsqueeze(0).to(device)
style_img_ink   = transform(Image.open("qianli.jpeg")).unsqueeze(0).to(device)
gen_img = content_img.clone().requires_grad_(True)

# VGG 特征提取器
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# ========== 层定义：内容层和风格层分开 ==========
content_layer = 'conv4_2'   # 只用 conv4_2 做内容
style_layers  = {
    'conv1_1' : {'oil' : 0.0, 'ink' : 1.0},
    'conv2_1' : {'oil' : 0.1, 'ink' : 0.9},
    'conv3_1' : {'oil' : 0.4, 'ink' : 0.6},
    'conv4_1' : {'oil' : 0.9, 'ink' : 0.1},
    'conv5_1' : {'oil' : 1.0, 'ink' : 0.0},
}

layer_map = {
    '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
    '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1',
}

def get_features(img_tensor, model):
    features = {}
    x = img_tensor
    for name, layer in model.named_children():
        x = layer(x)
        if name in layer_map:
            features[layer_map[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    f = tensor.view(b, c, h * w)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / (c * h * w)

# 预计算目标特征
content_feats = get_features(content_img, vgg)[content_layer]
style_feats_oil   = get_features(style_img_oil, vgg)
style_feats_ink   = get_features(style_img_ink, vgg)
style_grams_oil   = {l: gram_matrix(style_feats_oil[l]) for l in style_layers}
style_grams_ink   = {l: gram_matrix(style_feats_ink[l]) for l in style_layers}

# ========== TV 直接作用在像素上 ==========
def tv_loss(img):
    return (torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])) +
            torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])))

# ========== 损失（权重重新标定）==========
alpha = 1.0          # 内容权重
beta  = 1e6          # 风格权重（可调，越大风格越浓）
gamma = 1e-6         # TV 权重（作用在像素上，数值很大，所以要用小权重）

def compute_loss(gen_img, gen_features):
    c_loss = torch.mean((gen_features[content_layer] - content_feats) ** 2)

    s_loss = 0.0
    for l in style_layers:
        g = gram_matrix(gen_features[l])
        w_oil = style_layers[l]['oil']
        w_ink = style_layers[l]['ink']
        mixed_style_grams =  w_oil * style_grams_oil[l] + w_ink * style_grams_ink[l]
        s_loss += torch.mean((g - mixed_style_grams) ** 2)
    s_loss /= len(style_layers)   # 除以层数，避免层数影响权重

    t_loss = tv_loss(gen_img)     # 直接传图像像素

    total = alpha * c_loss + beta * s_loss + gamma * t_loss
    return total, c_loss, s_loss, t_loss

# ========== 初始化 & 优化器 ==========
gen_img = content_img.clone().requires_grad_(True)
optimizer = torch.optim.LBFGS([gen_img], max_iter=20, history_size=50)

# ========== 训练循环 ==========
max_iter = 300

for step in range(max_iter):
    def closure():
        optimizer.zero_grad()
        gen_features = get_features(gen_img, vgg)
        total, c, s, t = compute_loss(gen_img, gen_features)
        total.backward()

        # 关键：每次更新后把像素 clamp 回合法范围，防止跑飞
        with torch.no_grad():
            gen_img.clamp_(-2.0, 2.5)

        if step % 50 == 0:
            print(f"Iter {step}: content={c.item():.2f}  style={s.item():.2f}  tv={t.item():.2f}")
        return total

    optimizer.step(closure)

# ========== 输出 ==========
final_img = unnormalize(gen_img.squeeze(0)).clamp(0, 1)
transforms.ToPILImage()(final_img).save("output.jpg")
print("图片已保存 output.jpg")