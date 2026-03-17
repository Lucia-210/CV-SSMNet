import torch

path = r"D:\zz\ReadPaper\TeacherZhang\PolSAR_Paper_Projects\CV-ASDF2Net\CV-ASDF2Net-main\Datasets\Barnaul\Barnaul\Test\0\0test_idx1_0.pth"
data = torch.load(path, map_location="cpu")

print(type(data))

if isinstance(data, dict):
    print(data.keys())
    for k, v in data.items():
        if torch.is_tensor(v):
            print(k, v.shape, v.dtype)
elif torch.is_tensor(data):
    print(data.shape, date.dtype)
else:
    print(data)