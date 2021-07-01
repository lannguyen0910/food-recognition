import torch
from torchvision import transforms as transforms
from .main_model import MODEL


def get_classification_predict(image_list):
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

    test_transforms = transforms.Compose([
                        transforms.Resize((256, 256)), 
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize
                    ]) 

    model_name = 'se_resnext101_32x4d'
    model = MODEL(
        num_classes= 500 , 
        senet154_weight = '', 
        multi_scale = True, 
        learn_region=True)

    # model = torch.nn.DataParallel(model)
    # vgg16 = model
    # vgg16.load_state_dict(torch.load('./model/ISIAfood500.pth'))
    # vgg16.cuda()
    model.eval()

    batch = [test_transforms(i) for i in image_list]
    input = torch.stack(batch)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    output, _, _= model(input_var)
    return output.data



    
