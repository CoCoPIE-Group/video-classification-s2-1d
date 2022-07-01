'''https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/split_readme.txt
There are totally 153 files in this folder, 
[action]_test_split[1-3].txt  corresponding to three splits reported in the paper.
The format of each file is
[video_name] [id]
The video is included in the training set if id is 1
The video is included in the testing set if id is 2
The video is not included for training/testing if id is 0
There should be 70 videos with id 1 , 30 videos with id 2 in each txt file.
'''
import os
import glob

os.makedirs('hmdb51_labels', exist_ok=True)
files = sorted(glob.glob(os.path.join('hmdbTrainTestlist', '*')))

labelfiles = {}
for split in [1, 2, 3]:
	labelfiles[split] = {}
	for set in ['train', 'test']:
		labelfiles[split][set] = open(os.path.join('hmdb51_labels', 'hmdb51_split{:1d}_{}.txt'.format(split, set)), 'w')

label_dict = {}
label_id = 0
class_file = open(os.path.join('hmdb51_labels', 'hmdb_labels.txt'), 'w')
for name in files:
	name_parts = name.split('_')
	label_name = '_'.join(name_parts[:-2]).split(os.sep)[-1]
	if label_name not in label_dict:
		label_id += 1
		label_dict[label_name] = label_id
		class_file.write('{} {}\n'.format(label_id, label_name))
	split = int(name_parts[-1].split('.')[0][-1])

	with open(name) as f:
		lines = f.read().strip().split('\n')
		for line in lines:
			line_parts = line.strip().split(' ')
			set_id = int(line_parts[1])
			if set_id == 1:
				labelfiles[split]['train'].write('{}/{} {}\n'.format(label_name, line_parts[0], label_dict[label_name]))
			elif set_id == 2:
				labelfiles[split]['test'].write('{}/{}\n'.format(label_name, line_parts[0]))

for split in [1, 2, 3]:
	for set in ['train', 'test']:
		labelfiles[split][set].close()