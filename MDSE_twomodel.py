import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import AttackMethods_SAGA as AttackMethods


#Main function
def main():

    modelDir1 = "./ann/ann_vgg16_cifar10_1.pth"

    SNNmodelDir2 = "./snn/snn_vgg16_cifar10_10_2.pth"


    # ==================================================
    dataset = 'CIFAR10'
    # dataset = 'CIFAR100'
    print('load model1 from: ', modelDir1)
    print('load model2 from: ', SNNmodelDir2)

    AttackMethods.MDSE_two(modelDir1, SNNmodelDir2, dataset)



if __name__ == '__main__':
    main()


