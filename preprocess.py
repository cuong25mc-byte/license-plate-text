import io, json, hashlib, random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import laplace
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold

# CẤU HÌNH
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'final_clean'
CSV_PATH = DATA_DIR / 'final_clean.csv'
OUT = BASE_DIR / 'processed'; FIG = OUT / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
IMG_H, IMG_W = 64, 160                   # đã chốt: 64x160 (xem thống kê canvas ở phần 4)
BATCH_SIZE, NUM_WORKERS = 64, 0          # NUM_WORKERS=0: tránh lỗi multiprocessing trên macOS/notebook
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def save_fig(name):
    plt.savefig(FIG / name, dpi=120, bbox_inches='tight'); plt.close()

def show(images, titles, name):          # lưới 2x8, images là mảng 2D giá trị [0, 1]
    fig, axes = plt.subplots(2, 8, figsize=(20, 4))
    for ax in axes.flat: ax.axis('off')
    for ax, im, t in zip(axes.flat, images, titles):
        ax.imshow(im, cmap='gray', vmin=0, vmax=1); ax.set_title(t, fontsize=8)
    save_fig(name)

class ResizeWithPadding:                 # giữ tỉ lệ ảnh, căn giữa, đệm trắng cho đủ IMG_W x IMG_H
    def __call__(self, img):
        w, h = img.size
        s = min(IMG_W / w, IMG_H / h)
        nw, nh = round(w * s), round(h * s)
        canvas = Image.new('L', (IMG_W, IMG_H), 255)
        canvas.paste(img.resize((nw, nh), Image.Resampling.BICUBIC), ((IMG_W - nw) // 2, (IMG_H - nh) // 2))
        return canvas

# ĐỌC DỮ LIỆU VÀ KIỂM TRA ĐỘ SẠCH 
images = {}                              # tên file -> đường dẫn thật (không phụ thuộc cấu trúc thư mục)
for p in DATA_DIR.rglob('*'):
    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        images[p.name.lower()] = p
        images.setdefault(p.stem.lower(), p)    # phòng khi CSV ghi tên file không có đuôi

df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
df = df.rename(columns={'images': 'file', 'image': 'file', 'path': 'file', 'labels': 'label'})[['file', 'label']]
df['label'] = df['label'].str.strip().str.upper()
df['path'] = [images.get(Path(f.strip()).name.lower()) or images.get(Path(f.strip()).stem.lower()) for f in df['file']]

def inspect(path):
    """Đọc ảnh 1 lần: kích thước, hash, độ sáng, tương phản, độ mờ (phương sai Laplacian)."""
    try:
        data = Path(path).read_bytes()
        img = Image.open(io.BytesIO(data)); img.load()
        g = np.asarray(img.convert('L').resize((128, 64)), dtype=np.float32)   # 128x64 cố định, KHÔNG đổi theo IMG_W: giữ thang đo giống lúc lọc ngưỡng
        return (img.width, img.height, hashlib.md5(data).hexdigest(), g.mean(), g.std(), laplace(g, mode='mirror').var())
    except Exception:
        return (0, 0, None, 0, 0, 0)     # file thiếu hoặc hỏng

cols = ['width', 'height', 'hash', 'brightness', 'contrast', 'blur']
df = pd.concat([df, pd.DataFrame([inspect(p) for p in df['path']], columns=cols, index=df.index)], axis=1)
df['aspect_ratio'] = df['width'] / df['height'].clip(lower=1)
print('Số ảnh:', len(df))

# Mỗi điều kiện phải đúng với TẤT CẢ ảnh. Số ảnh vi phạm phải bằng 0.
checks = {
    'file đọc được':             df['width'] > 0,
    'nhãn đúng [A-Z0-9]{6}':     df['label'].str.fullmatch(r'[A-Z0-9]{6}'),
    'width >= 32, height >= 16': (df['width'] >= 32) & (df['height'] >= 16),
    'tỉ lệ khung [0.3, 5.0)':    (df['aspect_ratio'] >= 0.3) & (df['aspect_ratio'] < 5.0),
    'độ sáng [30, 225)':         (df['brightness'] >= 30) & (df['brightness'] < 225),
    'tương phản >= 8':           df['contrast'] >= 8,
    'độ mờ >= 20':               df['blur'] >= 20,
    'không trùng ảnh':           ~df['hash'].duplicated(keep=False),
}
for name, ok in checks.items():
    print(f"{'PASS' if ok.all() else 'FAIL'}  {name:<28} vi phạm: {(~ok).sum()}")
failed = [n for n, ok in checks.items() if not ok.all()]
assert not failed, f'Dữ liệu chưa sạch, không đạt: {failed}'

metrics = ['width', 'height', 'aspect_ratio', 'brightness', 'contrast', 'blur']
summary = df[metrics].describe().round(1)
print(summary); summary.to_csv(OUT / 'quality_summary.csv')

# Kiểm tra bằng mắt: 16 ảnh mờ nhất còn lại (ảnh gốc, trước khi resize)
gray = lambda p: np.asarray(Image.open(p).convert('L')) / 255
d = df.nsmallest(16, 'blur')
show([gray(p) for p in d['path']], [f'{l} blur={b:.0f}' for l, b in zip(d['label'], d['blur'])], 'blurriest.png')
df = df.reset_index(drop=True)

# CHIA TRAIN / VAL / TEST (80/10/10) 
# Cùng nhãn không nằm ở 2 tập (group), tỉ lệ 3 mẫu biển số giống nhau ở mọi tập (stratify)
df['pattern'] = df['label'].str.replace('[A-Z]', 'A', regex=True).str.replace('[0-9]', '9', regex=True)
folds = [idx for _, idx in StratifiedGroupKFold(10, shuffle=True, random_state=SEED)
         .split(df, df['pattern'], groups=df['label'])]
test, val = df.iloc[folds[0]], df.iloc[folds[1]]
train = df.iloc[np.concatenate(folds[2:])]
splits = [('train', train), ('val', val), ('test', test)]
for name, d in splits:
    print(f'{name:<5} {len(d):>6} ảnh')

# VOCAB VÀ MÃ HÓA NHÃN 
PAD, SOS, EOS = 0, 1, 2
itos = ['<pad>', '<sos>', '<eos>'] + sorted(set(''.join(df['label'])))    # id -> ký tự
stoi = {c: i for i, c in enumerate(itos)}                                 # ký tự -> id
MAX_LEN = int(df['label'].str.len().max()) + 2                            # +sos +eos

def encode(label):                       # "AB1234" -> [sos, A, B, 1, 2, 3, 4, eos]
    return torch.tensor([SOS] + [stoi[c] for c in label] + [EOS])

def decode(ids):                         # bỏ pad/sos, dừng ở eos
    out = []
    for i in ids:
        i = int(i)
        if i == EOS: break
        if i > EOS: out.append(itos[i])
    return ''.join(out)

# Kiểm tra vocab: không trùng ký tự, mỗi ký tự có mặt trong train
char_count = Counter(''.join(train['label']))
assert len(set(itos)) == len(itos), 'vocab có ký tự trùng'
assert all(c in char_count for c in itos[3:]), 'có ký tự không xuất hiện trong train'
print(f'Vocab: {len(itos) - 3} ký tự + 3 token đặc biệt | MAX_LEN = {MAX_LEN}')
print('3 ký tự hiếm nhất trong train:', char_count.most_common()[-3:])

# TRANSFORM VÀ DATASET
# Thống kê cho kích thước đã chốt: % canvas được biển số chiếm, % ảnh bị chiều rộng giới hạn (ký tự nhỏ đi)
s = np.minimum(IMG_W / df['width'], IMG_H / df['height'])
print(f"Canvas dùng {(df['width'] * df['height'] * s**2 / (IMG_W * IMG_H)).mean():.0%}"
      f" | ảnh bị chiều rộng giới hạn {(IMG_W / df['width'] < IMG_H / df['height']).mean():.0%}")

# MEAN/STD tính từ 2000 ảnh train sau ResizeWithPadding (chưa chuẩn hóa)
resize_pad = ResizeWithPadding()
sample = torch.stack([T.ToTensor()(resize_pad(Image.open(p).convert('L')))
                      for p in train['path'].sample(min(2000, len(train)), random_state=SEED)])
MEAN, STD = sample.mean().item(), sample.std().item()
print(f'Train MEAN: {MEAN:.4f}, Train STD: {STD:.4f}')
assert STD > 0

transform = T.Compose([resize_pad, T.ToTensor(), T.Normalize((MEAN,), (STD,))])   # không augmentation

class PlateDataset(Dataset):
    def __init__(self, data, transform):
        self.paths, self.labels, self.tf = list(data['path']), list(data['label']), transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('L')), encode(self.labels[i])

train_loader = DataLoader(PlateDataset(train, transform), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(PlateDataset(val, transform), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(PlateDataset(test, transform), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# KIỂM TRA CUỐI VÀ LƯU 
# 5a. Chia tập: không mất ảnh, không trùng ảnh/nhãn giữa các tập, tỉ lệ mẫu biển số đồng đều
assert sum(len(d) for _, d in splits) == len(df), 'chia tập làm mất hoặc nhân đôi ảnh'
for (n1, d1), (n2, d2) in [(splits[0], splits[1]), (splits[0], splits[2]), (splits[1], splits[2])]:
    assert not set(d1['path'].astype(str)) & set(d2['path'].astype(str)), f'{n1}/{n2} trùng ảnh'
    assert not set(d1['label']) & set(d2['label']), f'{n1}/{n2} trùng nhãn'
share = pd.concat({n: d['pattern'].value_counts(normalize=True) for n, d in splits}, axis=1)
print(share.round(3))
assert (share.max(axis=1) - share.min(axis=1)).max() < 0.05, 'tỉ lệ mẫu biển số lệch giữa các tập'

# 5b. Encode/decode: mọi nhãn mã hóa rồi giải mã phải về đúng nhãn cũ
assert all(decode(encode(l)) == l for l in df['label']), 'encode/decode không khớp'

# 5c. Tensor: kiểm tra 1 batch của mỗi loader + lưu ảnh batch (đúng thứ tự model nhìn thấy, kèm nhãn giải mã)
for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
    x, y = next(iter(loader))
    assert x.shape[1:] == (1, IMG_H, IMG_W) and x.dtype == torch.float32, 'sai shape/dtype ảnh'
    assert torch.isfinite(x).all(), 'ảnh có NaN/Inf'
    assert y.shape[1] == MAX_LEN and y.dtype == torch.long, 'sai shape/dtype nhãn'
    assert (y[:, 0] == SOS).all() and (y[:, -1] == EOS).all() and y.max() < len(itos), 'nhãn sai token'
    print(f'{name:<5} ảnh {tuple(x.shape)} | nhãn {tuple(y.shape)} | mean {x.mean():.2f} std {x.std():.2f} | ví dụ: {decode(y[0])}')
    show([(im[0] * STD + MEAN).clamp(0, 1).numpy() for im in x[:16]], [decode(t) for t in y[:16]], f'batch_{name}.png')

# Chỉ lưu sau khi mọi kiểm tra đạt: CSV (đường dẫn tương đối) + meta.json cho notebook train
for name, d in splits:
    pd.DataFrame({'path': d['path'].map(lambda p: p.relative_to(BASE_DIR).as_posix()), 'label': d['label']}) \
        .to_csv(OUT / f'{name}.csv', index=False)
meta = dict(itos=itos, pad=PAD, sos=SOS, eos=EOS, max_len=MAX_LEN, img_h=IMG_H, img_w=IMG_W, channels=1,
            mean=MEAN, std=STD, resize='letterbox_fill255', augmentation=False, seed=SEED)
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2))
print('XONG: dữ liệu sẵn sàng. File và biểu đồ đã lưu trong', OUT.resolve())