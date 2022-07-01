 import os
import yaml
from .opt_main import main as t_admm
from .options import parse_args as p_admm
# from .pytorch.train_mask_t import main as t_mask, parse_args as p_mask
# from .pytorch.test_mask_t import main as test_mask, parse_args as test_parse


def main():
    def print_info(info):
        print('---------------------------------------------------')
        print(info)
        print('---------------------------------------------------')

    str1 = input("Please enter the accuracy requirement (default 0.8):")
    str2 = input("Please enter the latency requirement (default none):")
    str3 = input("Please enter the data location (default '/raid10/ms/dataset'):")

    str4 = input("Please enter the architecture name (default c3d):")
    # str4 = input("Please enter the architecture name (default c3d):")

    # acc_req = requirement['accuracy']
    # speed_req = requirement['inference_time']

    if str1 != '':
        acc_req = float(str1)
    else:
        acc_req = 0.8
    if str2 != '':
        speed_req = float(str2)
    else:
        speed_req = 1000000
    if str3 != '':
        data_location = str3
    else:
        data_location = '/raid10/ms/dataset'
        # data_location = './dataset'
    if str4 != '':
        arch = str4
    else:
        arch = 'c3d'

    print('The accuracy requirement is {}'.format(acc_req))
    print('The speed requirement is {}'.format(speed_req))
    print('The data location is {}'.format(data_location))
    print('The architecture is {}'.format(arch))

    # ckpts_prune = 'ckpts/ckpt_a'
    # ckpts_retrain = 'ckpts/ckpt_m'
    ckpts_final = 'ckpts/ckpt_f'
    # if not os.path.exists(ckpts_prune):
    #     os.makedirs(ckpts_prune)
    # if not os.path.exists(ckpts_retrain):
    #     os.makedirs(ckpts_retrain)
    if not os.path.exists(ckpts_final):
        os.makedirs(ckpts_final)

    if arch == 'c3d':
        if acc_req > 0.8:
            prune_file = 'c3d_2.53x_v2'
            flops_prune_rate='2.53x'
        else:
            prune_file = 'c3d_3.57x'
            flops_prune_rate = '3.57x'
    elif arch== 'r2+1d' or 'r2+1d-pretrained':
        if acc_req >= 0.92:
            prune_file = 'r2+1d-pretrained_2.56x'
            flops_prune_rate = '2.56x'
        else:
            prune_file = 'r2+1d-pretrained_3.20x'
            flops_prune_rate = '3.20x'
    elif arch== 's3d':
        prune_file = 's3d_2.09x'
        flops_prune_rate = '2.09x'
    else:
        print('model arch not found')

    cmd_prune = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python ./ModelOpt/rt3d_pruning/opt_main.py --multi-gpu --dataset ucf101 --sparsity-type blk-kgs --connectivity-block-size 8 4 --admm --rho 0.0001 --rho-num 4 --epoch 50 --lr 5e-4 --optmzr sgd --log-interval 50 --smooth --smooth-eps 0.1 --data_location=\'{}\''.format(
        data_location)

    cmd_retrain = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python ./ModelOpt/rt3d_pruning/opt_main.py --multi-gpu --dataset ucf101 --sparsity-type blk-kgs --connectivity-block-size 8 4 --combine-progressive --masked-retrain --rho 0.0001 --rho-num 4 --epoch 130 --lr 5e-4 --optmzr sgd --log-interval 50 --warmup --warmup-lr 1e-5 --lr-scheduler cosine --distill --teacharch c3d --teacher-path checkpoint/ucf101_c3d_transfer_epoch-14_top1-81.571.pt #--smooth --smooth-eps 0.1 --data_location=\'{}\''.format(
        data_location)

    # cmd_test_prune = 'python pytorch/test_mask.py --model_dir=\'{}\'/best_checkpoints --data_location=\'{}\' --sparsity-type=\'{}\''.format(
    #     ckpts_prune, data_location, sparsity)
    # cmd_test_retrain = 'python pytorch/test_mask.py --model_dir=\'{}\'/best_checkpoints --data_location=\'{}\' --sparsity-type=\'{}\''.format(
    #     ckpts_retrain, data_location, sparsity)

    cmd_prune_new = cmd_prune + ' --arch {} --config-file=\'{}\''.format(arch, prune_file)  # must leave a space at the beginning
    cmd_retrain_new = cmd_retrain + ' --arch {} --config-file=\'{}\''.format(arch, prune_file)

    print_info('the estimated FLOPs pruning rate is {}.'.format(flops_prune_rate))

    print_info('start pruning')

    print(cmd_prune_new) #Todo: test package

    parser_admm = p_admm()
    args_admm = parser_admm.parse_args(['--data_location', data_location, '--admm', '--epoch', '50', '--arch', arch, '--config-file', prune_file]) # note the square bracket
    # args_admm = parser_admm.parse_args(['--data_location', data_location, '--epoch', '50'])
    t_admm(args_admm)
    # os.system(cmd_prune_new)



    print_info('finish prune, start retrain.')

    print(cmd_retrain_new)  #Todo: test package

    parser_mask = p_admm()
    args_mask = parser_mask.parse_args(['--data_location', data_location, '--combine-progressive', '--masked-retrain',  '--epoch', '130', '--warmup',  '--arch', arch, '--config-file', prune_file]) # note the square bracket
    t_admm(args_mask)
    # os.system(cmd_retrain_new)





    print_info('you can find the model at checkpoint folder')



# compare and obtain the best accuracy
# acc = np.zeros(4)
# for i in range(4):
#	acc[i] = open("ckpts/ckpt_m_{}/acc.txt".format(i)).read().float()

# maxlocation = np.argmax(acc)

# print_info('you can find the model at ckpts/ckpt_m_{} with accuracy {}'.format(maxlocation, acc[maxlocation]))

#######################################################################################################
#     for i in range(4):
#         print_info('start prune ratio case {}.'.format(i))
#
#         ckpts_prune_new = 'ckpts/ckpt_a_{}'.format(i)
#         ckpts_retrain_new = 'ckpts/ckpt_m_{}'.format(i)
#         ckpts_final_new = 'ckpts/ckpt_f_{}'.format(i)
#         if not os.path.exists(ckpts_prune_new):
#             os.makedirs(ckpts_prune_new)
#         if not os.path.exists(ckpts_retrain_new):
#             os.makedirs(ckpts_retrain_new)
#         if not os.path.exists(ckpts_final_new):
#             os.makedirs(ckpts_final_new)
#
#         copy_ckpts_original = 'cp ckpts/ckpts_original/* {}'.format(ckpts_prune_new)
#         os.system(copy_ckpts_original)
#         print_info('obtain original model')
#
#         print_info('start prune.')
#
#         cmd_prune_new = cmd_prune + ' --config-file=\'ratio_{}\''.format(i)
#         cmd_retrain_new = cmd_retrain + ' --config-file=\'ratio_{}\''.format(i)
#         cmd_test_prune_new = cmd_test_prune + ' --config-file=\'ratio_{}\''.format(i)
#         cmd_test_retrain_new = cmd_test_retrain + ' --config-file=\'ratio_{}\''.format(i)
#         print(cmd_prune_new)
#
#         #  parser_admm = p_admm()
#         # args_admm = parser_admm.parse_args(['--model_dir', ckpts_prune_new, '--data_location', data_location, '--sparsity-type', sparsity, '--config-file', 'ratio_{}'.format(i)]) # note the square bracket
#         #  t_admm(args_admm)
#         # os.system(cmd_prune_new)
#
#         print(cmd_test_prune_new)
#         # os.system(cmd_test_prune_new)
#         # parser_test = test_parse()
#         #   args_test = parser_test.parse_args(['--model_dir', ckpts_prune_new, '--data_location', data_location, '--sparsity-type', sparsity, '--config-file', 'ratio_{}'.format(i)]) # note the square bracket
#         #  test_mask(args_test)
#         print_info('finish prune with the above accuracy')
#
#         copy_ckpts_prune = 'cp {}/best_checkpoints/*  {}'.format(ckpts_prune_new, ckpts_retrain_new)
#         os.system(copy_ckpts_prune)
#         print_info('obtain pruned model')
#
#         print_info('finish prune, start retrain.')
#
#         print(cmd_retrain_new)
#         # os.system(cmd_retrain_new)
#         #    parser_mask = p_mask()
#         #   args_mask = parser_mask.parse_args(['--model_dir', ckpts_retrain_new, '--data_location', data_location, '--sparsity-type', sparsity, '--config-file', 'ratio_{}'.format(i)]) # note the square bracket
#         # to overwrite default, use parser.parse_args(['--input', 'foobar.txt'])
#         #  t_mask(args_mask)
#
#         print_info('finish retrain.')
#
#         print(cmd_test_retrain_new)
#         # os.system(cmd_test_retrain_new)
#         #  parser_test = test_parse()
#         #  args_test = parser_test.parse_args(['--model_dir', ckpts_retrain_new, '--data_location', data_location, '--sparsity-type', sparsity, '--config-file', 'ratio_{}'.format(i)]) # note the square bracket
#         #  test_mask(args_test)
#         print_info('finish retrain with the above accuracy')
#
#         copy_ckpts_prune = 'cp {}/best_checkpoints/*  {}'.format(ckpts_retrain_new, ckpts_final_new)
#         os.system(copy_ckpts_prune)
#         print_info('obtain final model')
#
#         print_info('you can find the model at {}'.format(ckpts_final_new))
#     print_info('you can find the model with the best accuracy by comparing the above.')
#
#
# # compare and obtain the best accuracy
# # acc = np.zeros(4)
# # for i in range(4):
# #	acc[i] = open("ckpts/ckpt_m_{}/acc.txt".format(i)).read().float()
#
# # maxlocation = np.argmax(acc)
#
# # print_info('you can find the model at ckpts/ckpt_m_{} with accuracy {}'.format(maxlocation, acc[maxlocation]))

if __name__ == '__main__':
    main()







