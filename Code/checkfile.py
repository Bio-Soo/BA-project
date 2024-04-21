
path = 'my_model.pth'

model.load_state_dict(torch.load('model_weights.pth'))
model = torch.load('model.pth')

print(model)