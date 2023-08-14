import torch

print("STARTING", flush=True)

a = torch.tensor([1, 2, 3])
print(a, flush = True)

# Check if cuda is available:
print(torch.cuda.is_available(), flush = True)


a.to("cuda")
print(a, flush = True)

print("END")