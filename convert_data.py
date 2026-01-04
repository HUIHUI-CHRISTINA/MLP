import os
import numpy as np
import shutil

base_path = 'c:/Users/86159/OneDrive/桌面/s3net-master/data/QuickDraw414k/coordinate_files'
output_path = 'c:/Users/86159/OneDrive/桌面/s3net-master/data/quickdraw'

# 获取类别列表
categories = os.listdir(os.path.join(base_path, 'train'))
categories.sort()
print(f'Found {len(categories)} categories')

# 复制 theta.npy
shutil.copy('c:/Users/86159/OneDrive/桌面/s3net-master/src/theta.npy', os.path.join(output_path, 'theta.npy'))

# 创建 sketchrnn.txt
with open(os.path.join(output_path, 'sketchrnn.txt'), 'w') as f:
    for cat in categories:
        f.write(os.path.join(output_path, f'{cat}.npy') + '\n')

# 处理每个类别
for cat in categories:
    print(f'Processing {cat}')
    data = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split, cat)
        files = os.listdir(split_path)
        files = [f for f in files if f.endswith('.npy')]
        files.sort()
        samples = []
        for file in files:
            try:
                sample = np.load(os.path.join(split_path, file), allow_pickle=True, encoding='latin1')
                if sample.shape == (100, 4):
                    # Convert to SketchRNN format: (delta_x, delta_y, pen_state)
                    # Assuming sample is [x, y, p1, p2], where p1=1 for draw, p2=1 for lift
                    strokes = []
                    prev_x, prev_y = 0, 0
                    for i in range(len(sample)):
                        x, y, p1, p2 = sample[i]
                        delta_x = x - prev_x
                        delta_y = y - prev_y
                        if i == 0:
                            pen_state = 0  # start
                        elif p2 == 1:
                            pen_state = 1  # lift
                        else:
                            pen_state = 0  # draw
                        strokes.append([delta_x, delta_y, pen_state])
                        if p2 == 0:  # update prev only if not lift
                            prev_x, prev_y = x, y
                    # Remove trailing zeros or something, but keep as is for now
                    samples.append(np.array(strokes))
                else:
                    print(f'Skipping {file} with shape {sample.shape}')
            except Exception as e:
                print(f'Error loading {file}: {e}')
                continue
        if split == 'val':
            data['valid'] = np.array(samples, dtype=object)
        else:
            data[split] = np.array(samples, dtype=object)
    np.save(os.path.join(output_path, f'{cat}.npy'), data)

print('Done')