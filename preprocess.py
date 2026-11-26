# ===== PREPROCESS (chạy local): ảnh biển số -> (tensor ảnh, chuỗi ký tự) =====
# Cài thư viện (1 lần):  pip install torch torchvision scikit-learn scipy pandas pillow matplotlib
# Các bước: 0.Cấu hình  1.Đọc & kiểm tra độ sạch  2.Chia tập  3.Vocab  4.Transform & Dataset  5.Kiểm tra cuối
# %%
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

class ResizeWithPadding:
    def __init__(self, height=64, width=128, fill=255):
        self.height = height
        self.width = width
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.width / w,self.height / h)
        new_w = round(w * scale)
        new_h = round(h * scale)
        img = img.resize((new_w, new_h),Image.Resampling.BICUBIC)
        canvas = Image.new('L',(self.width, self.height),self.fill)
        left = (self.width - new_w) // 2
        top = (self.height - new_h) // 2
        canvas.paste(img, (left, top))
        return canvas

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'final_clean'
CSV_PATH = DATA_DIR / 'final_clean.csv'
IMG_DIR = DATA_DIR / 'images'

OUT = BASE_DIR / "processed"
OUT.mkdir(parents=True, exist_ok=True)
IMG_H = 64       
IMG_W = 128             
BATCH_SIZE = 64
NUM_WORKERS = 0                          # để 0 cho macOS/notebook, tránh lỗi multiprocessing
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
        g = np.asarray(img.convert('L').resize((128, 64)), dtype=np.float32)
        return (img.width, img.height, hashlib.md5(data).hexdigest(),
                g.mean(), g.std(), laplace(g, mode='mirror').var())
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
print(df[['width', 'height', 'aspect_ratio', 'brightness', 'contrast', 'blur']].describe().round(1))

# %% ---------- 2. CHIA TRAIN / VAL / TEST (80/10/10) ----------
# Cùng nhãn không nằm ở 2 tập (group), tỉ lệ 3 mẫu biển số giống nhau ở mọi tập (stratify)
df['pattern'] = df['label'].str.replace('[A-Z]', 'A', regex=True).str.replace('[0-9]', '9', regex=True)
folds = [idx for _, idx in StratifiedGroupKFold(10, shuffle=True, random_state=SEED)
         .split(df, df['pattern'], groups=df['label'])]
test, val = df.iloc[folds[0]], df.iloc[folds[1]]
train = df.iloc[np.concatenate(folds[2:])]
splits = [('train', train), ('val', val), ('test', test)]
for name, d in splits:
    export_df = pd.DataFrame({'file': d['path'].map(lambda p: Path(p).name), 'label': d['label']})
    export_df.to_csv(OUT / f'{name}.csv', index=False)
    print(f'{name:<5} {len(d):>6} ảnh')

# %% ---------- 3. VOCAB VÀ MÃ HÓA NHÃN ----------
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

# Kiểm tra vocab: không trùng ký tự, mỗi ký tự đủ nhiều mẫu trong train để model học được
char_count = Counter(''.join(train['label']))
assert len(set(itos)) == len(itos), 'vocab có ký tự trùng'
assert all(c in char_count for c in itos[3:]), 'có ký tự không xuất hiện trong train'
print(f'Vocab: {len(itos) - 3} ký tự + 3 token đặc biệt | MAX_LEN = {MAX_LEN}')
print('3 ký tự hiếm nhất trong train:', char_count.most_common()[-3:])

# %% ---------- 4. TRANSFORM VÀ DATASET ----------
# 4a. Chọn chiều rộng theo tỉ lệ khung trung vị của dữ liệu (bội số của 8) để ít làm méo ảnh
ar = df['aspect_ratio']
print(f'Tỉ lệ khung p10/p50/p90: {ar.quantile(.1):.2f}/{ar.median():.2f}/{ar.quantile(.9):.2f}')

# 4b. Tính MEAN và STD từ tập Train sau ResizeWithPadding, chưa augmentation# Tính MEAN và STD từ 2.000 ảnh Train sau ResizeWithPadding
resize_pad = ResizeWithPadding(height=IMG_H, width=IMG_W, fill=255)
sample_paths = train['path'].sample(min(2000, len(train)), random_state=SEED)
sample_tensors = [T.ToTensor()(resize_pad(Image.open(p).convert('L'))) for p in sample_paths]
sample_tensors = torch.stack(sample_tensors)
MEAN, STD = sample_tensors.mean().item(), sample_tensors.std().item()
print(f'Train MEAN: {MEAN:.4f}, Train STD: {STD:.4f}')
assert STD > 0

transform = T.Compose([resize_pad, T.ToTensor(), T.Normalize((MEAN,), (STD,))])

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

# Lưu mọi thông số để notebook train dùng lại đúng như vậy
json.dump(dict(itos=itos, pad=PAD, sos=SOS, eos=EOS, max_len=MAX_LEN,
               img_h=IMG_H, img_w=IMG_W, mean=MEAN, std=STD), open(OUT / 'meta.json', 'w'), indent=2)

# %% ---------- 5. KIỂM TRA CUỐI ----------
# 5a. Chia tập: không mất ảnh, không trùng ảnh/nhãn giữa các tập, tỉ lệ mẫu biển số đồng đều
assert sum(len(d) for _, d in splits) == len(df), 'chia tập làm mất hoặc nhân đôi ảnh'
for (n1, d1), (n2, d2) in [(splits[0], splits[1]), (splits[0], splits[2]), (splits[1], splits[2])]:
    assert not set(d1['path'].astype(str)) & set(d2['path'].astype(str)), f'{n1}/{n2} trùng ảnh'
    assert not set(d1['label']) & set(d2['label']), f'{n1}/{n2} trùng nhãn'
share = pd.concat({n: d['pattern'].value_counts(normalize=True) for n, d in splits}, axis=1)
print(share.round(3))
assert (share.max(axis=1) - share.min(axis=1)).max() < 0.05, 'tỉ lệ mẫu biển số lệch giữa các tập'

# 5b. Encode/decode: mọi nhãn mã hóa rồi giải mã phải về đúng nhãn cũ; meta.json đọc lại khớp
assert all(decode(encode(l)) == l for l in df['label']), 'encode/decode không khớp'
assert json.load(open(OUT / 'meta.json'))['itos'] == itos, 'meta.json không khớp vocab'

# 5c. Tensor: kiểm tra 1 batch của mỗi loader
for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
    x, y = next(iter(loader))
    assert x.shape[1:] == (1, IMG_H, IMG_W) and x.dtype == torch.float32, 'sai shape/dtype ảnh'
    assert torch.isfinite(x).all(), 'ảnh có NaN/Inf'
    assert y.shape[1] == MAX_LEN and y.dtype == torch.long, 'sai shape/dtype nhãn'
    assert (y[:, 0] == SOS).all() and (y[:, -1] == EOS).all() and y.max() < len(itos), 'nhãn sai token'
    print(f'{name:<5} ảnh {tuple(x.shape)} | nhãn {tuple(y.shape)} | mean {x.mean():.2f} std {x.std():.2f} | ví dụ: {decode(y[0])}')

config = {
    "image_height": IMG_H,
    "image_width": IMG_W,
    "channels": 1,
    "mean": MEAN,
    "std": STD,
    "augmentation": False,
    "resize": "ResizeWithPadding",
    "vocab_size": len(itos),
    "max_len": MAX_LEN,
    "seed": SEED
}

with open(OUT / "preprocess_config.json", "w") as f:
    json.dump(config, f, indent=4)

print('XONG: dữ liệu sẵn sàng. File đã lưu trong', OUT.resolve())