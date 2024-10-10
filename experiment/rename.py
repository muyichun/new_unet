import os

def rename_files_in_directory(directory, prefix='file_', start_index=1):
    # 获取目录下所有的文件名（不包括子目录）
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 为每个文件生成新的文件名并重命名
    for index, old_name in enumerate(files, start=start_index):
        # 生成新的文件名
        new_name = str(int(os.path.splitext(old_name)[0]) + 26001 ) + os.path.splitext(old_name)[1]

        # 构建完整的旧路径和新路径
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")


# 使用方法
directory_path = '/Users/muyichun/Desktop/3_210/实验数据/speckle_valid_128'  # 替换为你的文件夹路径
rename_files_in_directory(directory_path)