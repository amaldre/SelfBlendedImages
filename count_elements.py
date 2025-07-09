import os

def count(folder_path):
    return len(os.listdir(folder_path))

if __name__ == '__main__':
    SBI = r"/datasets/FaceForensics++/sbi/frames/954"
    print(count(SBI))
    ORIGINAL = r'/datasets/FaceForensics++/original_download/original_sequences/youtube'
    print(count(ORIGINAL))