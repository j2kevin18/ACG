import os
import tarfile
import pickle
import glob

# 搜索和解压缩的目录
search_extract_path = './fake_test'
# 存储jpg文件路径的pickle文件
pickle_path = 'fake.pickle'

# 搜索指定目录下的所有.tar.gz文件
tar_gz_files = glob.glob(os.path.join(search_extract_path, '*.tar.gz'))
print(tar_gz_files)

# 存储所有jpg文件路径的列表
all_jpg_files_paths = []

# 遍历找到的.tar.gz文件
for tar_gz_path in tar_gz_files:
    # 解压.tar.gz文件
    with tarfile.open(tar_gz_path) as tar_gz_ref:
        tar_gz_ref.extractall(search_extract_path)
    
    # 遍历解压缩后的目录，找到所有jpg文件
    jpg_files_paths = [os.path.join(root, file)
                       for root, dirs, files in os.walk(search_extract_path)
                       for file in files if file.endswith('.png')]
    
    # 将当前.tar.gz包中的jpg文件路径添加到总列表中
    all_jpg_files_paths.extend(jpg_files_paths)

# 将所有jpg文件路径列表保存为.pickle格式
with open(pickle_path, 'wb') as f:
    pickle.dump(all_jpg_files_paths, f)

print(f'所有jpg图片的文件路径已保存到 {pickle_path}')
