from os import listdir, makedirs
from os.path import join, isdir, isfile
from utils import read_script_files, read_script_file_data

# STA: speech text align

def data_prepare_for_STA(script_dir_path, align_data_dir_path, sub_dirs):
    # Prepare directory to save data
    for sub_dir in sub_dirs:
        if not isdir(join(align_data_dir_path, sub_dir)):
            makedirs(join(align_data_dir_path, sub_dir))

    # Process and save data
    script_files = read_script_files(script_dir_path)
    total_file_num = len(script_files)
    for sub_dir in sub_dirs:
        print('='*20)
        print('{} processing'.format(sub_dir))
        curr_file_num = 0
        for script_file in script_files:
            curr_file_num += 1
            print('Now processing {}/{}'.format(curr_file_num, total_file_num))

            line_num = 0
            curr_file_lines = read_script_file_data(script_dir_path, script_file)
            for line in curr_file_lines:
                line_num += 1
                save_file_name = 'output_{}_{}.lab'.format(script_file[ :4], line_num)
                save_file_name = join(join(align_data_dir_path, sub_dir), save_file_name)
                with open(save_file_name, 'w') as f:
                    f.write(line)

# # Prepare data for STA
# script_dir_path = './data/train_script'
# align_data_dir_path = './data/data_for_align'
# sub_dirs = ['Case_en-US_MALE', 'Case_en-US_FEMALE']
# data_prepare_for_STA(script_dir_path, align_data_dir_path, sub_dirs)

# Process STA in terminal (follow the link below)
# https://montreal-forced-aligner.readthedocs.io/en/latest/aligning.html
