import imp
from django.shortcuts import render

# Create your views here.

# def index(request):
#     return render(request,'project/index.html')





from django.shortcuts import render
from .models import Pic
import os
import glob
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from django.conf import settings
from .static.datasets.div2k.parameters import Div2kParameters 
from .static.models.srresnet import build_srresnet
from .static.models.srgan import build_discriminator
from .static.models.pretrained import pretrained_models
from .static.utils.prediction import get_sr_image
from .static.utils.config import config




def index(request):
    MEDIA_ROOT=settings.MEDIA_ROOT
    STATIC_ROOT=settings.STATIC_ROOT
    if request.method=='GET':
        print(MEDIA_ROOT)
        return render(request,'project/index.html')  
    else:
        dataset_key = "bicubic_x4"

        data_path = config.get("data_path", "") 

        div2k_folder = os.path.abspath(os.path.join(data_path, "./static/datasets/div2k"))

        dataset_parameters = Div2kParameters(dataset_key, save_data_directory=div2k_folder)

        print('hello')
        f=request.FILES['blurimage']
        
        obj=Pic(image=f)
        obj.save()
       
        
        # pa='\media'+'\weights\srgan_bicubic_x4\oo.jpg'
        # return render(request, 'hdimageconverter/home.html',{'context':pa})

        def test(model_key,path):
            

            def load_image(path):
                print('loaded weigkljskldjaldkjaskldjsalkjdklsdj')
                img = Image.open(path)
    
                was_grayscale = len(img.getbands()) == 1
    
                if was_grayscale or len(img.getbands()) == 4:
                    img = img.convert('RGB')
                return was_grayscale, np.array(img)
            print(MEDIA_ROOT)
            weights_directory = MEDIA_ROOT+'\weights\srgan_bicubic_x4'

            file_path =  MEDIA_ROOT+"\weights\srgan_bicubic_x4\generator.h5"

            # if not os.path.exists(file_path):
            #     os.makedirs(weights_directory, exist_ok=True)
    
            #     print("Couldn't find file: ", file_path, ", attempting to download a pretrained model")
    
            #     if model_key not in pretrained_models:
            #         print(f"Couldn't find pretrained model with key: {model_key}, available pretrained models: {pretrained_models.key()}")
            #     else:
            #         download_url = pretrained_models[model_key]
            #         file = file_path.split("/")[-1]
            #         tf.keras.utils.get_file(file, download_url, cache_subdir=weights_directory)
            model = build_srresnet(scale=dataset_parameters.scale)

            # os.makedirs(weights_directory, exist_ok=True)
            weights_file = f"{weights_directory}\generator.h5"

            model.load_weights(weights_file)
            print('loaded weights')
            results_path = MEDIA_ROOT+f"/output/{model_key}/"
            # os.makedirs(results_path, exist_ok=True)
            image_paths = glob.glob(path)
            print(image_paths)
            print('loaded weightsssssssssss')
            # for image_path in image_paths:
            #     print(image_path)
            #     was_grayscale, lr = load_image(image_path)
    
            #     sr = get_sr_image(model, lr)
            #     print('loa')
            #     if was_grayscale:
            #         sr = ImageOps.grayscale(sr)
    
            #     image_name = image_path.split("/")[-1]
            #     sr.save(f"{results_path}{image_name}" )
            # !zip -r images.zip output

            was_grayscale, lr = load_image(MEDIA_ROOT+path.replace('/media',''))

            print('getting')
    
            sr = get_sr_image(model, lr)
            print('loa')
            if was_grayscale:
                sr = ImageOps.grayscale(sr)
    
            image_name = path.split("/")[-1]
            sr.save(f"{results_path}{image_name}" )
            

        model_name = "srgan"
        model_key = f"{model_name}_{dataset_key}"
        path=obj.image.url
        test(model_key,path)
        print(MEDIA_ROOT)
        return render(request, 'project/home.html',{'context':str("media/output/srgan_bicubic_x4/"+obj.image.name)})
        
